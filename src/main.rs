use axum::{
    response::{IntoResponse, Html, Json},
    routing::{get, post},
    extract::{State, Form, Path},
    Router,
    http::StatusCode,
};
use serde_json::json;

macro_rules! log {
    ($($arg:tt)*) => {{
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        println!("[{}] {}", timestamp, format!($($arg)*));
    }};
}
#[derive(Debug)]
#[allow(dead_code)]
enum AppError {
    SystemError(String),
    AuthError(String),
    NotFound(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            AppError::SystemError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            AppError::AuthError(msg) => (StatusCode::UNAUTHORIZED, msg),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
        };
        
        (status, Html(message)).into_response()
    }
}

impl From<serde_json::Error> for AppError {
    fn from(err: serde_json::Error) -> Self {
        AppError::SystemError(format!("JSON error: {}", err))
    }
}

impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> Self {
        AppError::SystemError(format!("IO error: {}", err))
    }
}

impl From<tokio::sync::TryLockError> for AppError {
    fn from(_: tokio::sync::TryLockError) -> Self {
        AppError::SystemError("Lock acquisition failed".to_string())
    }
}

impl From<Box<dyn std::error::Error>> for AppError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        AppError::SystemError(err.to_string())
    }
}
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::{SystemTime, Duration};
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use uuid::Uuid;
use bcrypt::{hash, verify, DEFAULT_COST};

#[derive(Serialize)]
struct KnowledgeGraphData {
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
}

#[derive(Serialize)]
struct GraphNode {
    id: String,
    label: String,
    level: u8,
    mastery: f32,
}

#[derive(Serialize)]
struct GraphEdge {
    source: String,
    target: String,
}

// ---- Core Data Structures ----

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Goal {
    pub id: Uuid,
    pub description: String,
    pub criteria: Vec<String>,    // Specific criteria for goal completion
    pub tags: Vec<String>,
    pub status: GoalStatus,
    pub created_at: DateTime<Utc>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum GoalStatus {
    Discovery,           // Still being refined
    Active,             // Currently being worked on
    Completed,          // Goal has been achieved
    Archived,           // No longer active
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SpacedRepetitionInfo {
    pub last_reviewed: DateTime<Utc>,
    pub next_review: DateTime<Utc>,
    pub interval: i32,  // Days until next review
    pub ease_factor: f32,  // Multiplier for interval adjustments (default 2.5)
    pub consecutive_correct: i32,
}

#[derive(Clone, Serialize, Deserialize, PartialEq,Debug)]
pub struct Card {
    pub id: Uuid,
    pub goal_id: Uuid,
    pub question: String,
    pub answer: String,
    pub context: String,         // Required context/explanation
    pub difficulty: u8,          // 1-5 scale
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub review_count: u32,
    pub success_rate: f32,
    pub spaced_rep: SpacedRepetitionInfo,
    pub prerequisites: Vec<Uuid>,  // Cards that should be learned first
    pub last_reviewed: Option<DateTime<Utc>>
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LearningPoint {
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub mastery_level: f32,  // 0-1 scale indicating understanding
}

impl std::fmt::Display for LearningPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.content)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Discussion {
    pub id: Uuid,
    pub card_id: Uuid,
    pub user_response: String,
    pub correctness_score: f32,  // 0-1 scale
    pub critique: String,
    pub learning_points: Vec<LearningPoint>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TagPerformance {
    pub tag: String,
    pub total_attempts: u32,
    pub success_count: u32,
    pub failure_count: u32,
    pub average_score: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UserProgress {
    pub total_cards_reviewed: u64,
    pub total_study_sessions: u64,
    pub tag_performance: Vec<TagPerformance>,
    pub active_goals: Vec<Uuid>,
    pub completed_goals: Vec<Uuid>,
    pub last_session: Option<DateTime<Utc>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CurriculumModule {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub order: i32,
    pub estimated_hours: f32,
    pub topics: Vec<CurriculumTopic>,
    pub dependencies: Vec<Uuid>,  // Prerequisites modules
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CurriculumTopic {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub cards: Vec<Uuid>,
    pub order: i32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct LearningSystem {
    #[serde(default = "default_version")]
    version: u32,
    #[serde(default)]
    pub goals: Vec<Goal>,
    #[serde(default)]
    pub cards: Vec<Card>,
    #[serde(default)]
    pub discussions: Vec<Discussion>,
    pub progress: UserProgress,
    #[serde(default)]
    pub curriculum: Vec<CurriculumModule>,
}

fn default_version() -> u32 { 1 }

#[derive(Debug)]
struct LoginAttempt {
    attempts: u32,
    last_attempt: SystemTime,
}

#[derive(Deserialize)]
struct LoginForm {
    password: String,
}

struct AppState {
    learning_system: LearningSystem,
    login_attempts: HashMap<String, LoginAttempt>,
    password_hash: String,
}


// ---- OpenAI API Structures ----

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub n: Option<u32>,
}

#[derive(Deserialize, Debug)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Option<ChatUsage>,
}

#[derive(Deserialize, Debug)]
pub struct ChatChoice {
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
    pub index: Option<u32>,
}

#[derive(Deserialize, Debug)]
pub struct ChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ---- Implementation ----

#[derive(Deserialize)]
struct CurriculumUpdate {
    modules: Vec<CurriculumModule>,
}

async fn update_curriculum(
    State(state): State<Arc<Mutex<AppState>>>,
    Path(_goal_id): Path<Uuid>,
    Json(update): Json<CurriculumUpdate>,
) -> Result<impl IntoResponse, AppError> {
    let mut state = state.lock().map_err(|_| AppError::SystemError("Lock error".to_string()))?;
    
    // Validate that all modules have valid prerequisites
    for module in &update.modules {
        for prereq_id in &module.dependencies {
            if !update.modules.iter().any(|m| m.id == *prereq_id) {
                return Err(AppError::SystemError("Invalid prerequisite reference".to_string()));
            }
        }
    }
    
    // Update curriculum
    state.learning_system.curriculum = update.modules;
    
    // Save changes
    if let Err(e) = state.learning_system.save("learning_system.json") {
        log!("ERROR: Failed to save curriculum changes: {}", e);
        return Err(AppError::SystemError("Failed to save changes".to_string()));
    }
    
    Ok(Json(json!({ "status": "success" })))
}

impl LearningSystem {
    fn adjust_card_difficulty(&mut self, card_id: Uuid, performance_history: &[f32]) -> Result<(), Box<dyn Error>> {
        let card = self.cards.iter_mut()
            .find(|c| c.id == card_id)
            .ok_or("Card not found")?;
        
        // Calculate trend from recent performances
        let trend = if performance_history.len() >= 3 {
            let recent = &performance_history[performance_history.len()-3..];
            (recent[2] - recent[0]) / 2.0
        } else {
            0.0
        };

        // Adjust difficulty based on performance trend
        if trend > 0.2 && card.difficulty < 5 {
            card.difficulty += 1;
        } else if trend < -0.2 && card.difficulty > 1 {
            card.difficulty -= 1;
        }

        Ok(())
    }

    fn generate_practice_session(&self, goal_id: Uuid, duration_minutes: u32) -> Vec<&Card> {
        let mut available_time = duration_minutes;
        let mut session_cards = Vec::new();
        
        // Get due cards first, excluding recently successful ones
        let mut due_cards: Vec<&Card> = self.get_due_cards().into_iter()
            .filter(|c| {
                c.goal_id == goal_id && 
                // Only include if either:
                // 1. Card hasn't been reviewed recently, or
                // 2. Last review wasn't successful (success_rate < 0.8)
                match c.last_reviewed {
                    Some(last_review) => {
                        let hours_since_review = (Utc::now() - last_review).num_hours();
                        hours_since_review > 24 || c.success_rate < 0.8
                    },
                    None => true
                }
            })
            .collect();
        
        // Sort by overdue duration
        due_cards.sort_by(|a, b| {
            b.spaced_rep.next_review.cmp(&a.spaced_rep.next_review)
        });

        // Add due cards until we fill half the session time
        while available_time > duration_minutes / 2 && !due_cards.is_empty() {
            if let Some(card) = due_cards.pop() {
                session_cards.push(card);
                available_time -= 2; // Assume 2 minutes per card
            }
        }

        // Fill remaining time with cards that need reinforcement
        let weak_cards: Vec<&Card> = self.cards.iter()
            .filter(|c| {
                c.goal_id == goal_id 
                && c.success_rate < 0.7 
                && !session_cards.contains(c)
                // Add time-based filter here too
                && match c.last_reviewed {
                    Some(last_review) => {
                        (Utc::now() - last_review).num_hours() > 12
                    },
                    None => true
                }
            })
            .collect();

        for card in weak_cards {
            if available_time < 2 { break; }
            session_cards.push(card);
            available_time -= 2;
        }

        session_cards
    }

    fn generate_knowledge_graph(&self, goal_id: Uuid) -> KnowledgeGraphData {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut processed_concepts = std::collections::HashSet::new();

        // Add nodes for each card
        for card in &self.cards {
            if card.goal_id == goal_id {
                for tag in &card.tags {
                    if processed_concepts.insert(tag) {
                        // Calculate mastery level for this concept
                        let related_cards: Vec<&Card> = self.cards.iter()
                            .filter(|c| c.tags.contains(tag))
                            .collect();
                        
                        let mastery = related_cards.iter()
                            .map(|c| c.success_rate)
                            .sum::<f32>() / related_cards.len() as f32;

                        nodes.push(GraphNode {
                            id: tag.clone(),
                            label: tag.clone(),
                            level: card.difficulty,
                            mastery,
                        });
                    }
                }

                // Add edges for prerequisites
                for prereq_id in &card.prerequisites {
                    if let Some(prereq_card) = self.cards.iter().find(|c| c.id == *prereq_id) {
                        for prereq_tag in &prereq_card.tags {
                            for tag in &card.tags {
                                edges.push(GraphEdge {
                                    source: prereq_tag.clone(),
                                    target: tag.clone(),
                                });
                            }
                        }
                    }
                }
            }
        }

        KnowledgeGraphData { nodes, edges }
    }

    fn calculate_next_review(&self, card: &Card, performance: f32) -> SpacedRepetitionInfo {
        let mut spaced_rep = card.spaced_rep.clone();
        
        // SuperMemo 2 algorithm with performance-based adjustments
        if performance >= 0.6 {  // Lower threshold for "correct" answers
            spaced_rep.consecutive_correct += 1;
            
            // Scale interval increase based on performance
            let performance_multiplier = 0.5 + (performance * 0.5);  // Range: 0.5-1.0
            
            if spaced_rep.consecutive_correct == 1 {
                spaced_rep.interval = 1;
            } else if spaced_rep.consecutive_correct == 2 {
                spaced_rep.interval = (6.0 * performance_multiplier) as i32;
            } else {
                let base_interval = (spaced_rep.interval as f32) * spaced_rep.ease_factor;
                spaced_rep.interval = (base_interval * performance_multiplier) as i32;
            }
            
            // Adjust ease factor based on performance
            let ease_adjustment = match performance {
                x if x >= 0.9 => 0.15,  // Excellent performance
                x if x >= 0.8 => 0.10,  // Good performance
                x if x >= 0.7 => 0.05,  // Decent performance
                _ => 0.0,               // Barely passing
            };
            spaced_rep.ease_factor += ease_adjustment;
        } else {
            // Reset for failed reviews, with severity based on performance
            spaced_rep.consecutive_correct = 0;
            spaced_rep.interval = if performance < 0.3 { 1 } else { 2 };  // Shorter reset for near-misses
            spaced_rep.ease_factor -= 0.2 * (0.6 - performance);  // More penalty for worse performance
        }
        
        // Ensure bounds
        spaced_rep.ease_factor = spaced_rep.ease_factor.max(1.3).min(2.5);
        spaced_rep.interval = spaced_rep.interval.max(1);
        
        // Update timestamps
        spaced_rep.last_reviewed = Utc::now();
        spaced_rep.next_review = Utc::now() + chrono::Duration::days(spaced_rep.interval as i64);
        
        spaced_rep
    }
    pub fn get_due_cards(&self) -> Vec<&Card> {
        self.cards.iter()
            .filter(|card| {
                card.spaced_rep.next_review <= Utc::now()
            })
            .collect()
    }

    fn identify_weak_topics(&self) -> Vec<String> {
        self.progress.tag_performance.iter()
            .filter(|perf| perf.average_score < 0.7)
            .map(|perf| perf.tag.clone())
            .collect()
    }

    pub fn new() -> Self {
        LearningSystem {
            goals: Vec::new(),
            cards: Vec::new(),
            discussions: Vec::new(),
            progress: UserProgress {
                total_cards_reviewed: 0,
                total_study_sessions: 0,
                tag_performance: Vec::new(),
                active_goals: Vec::new(),
                completed_goals: Vec::new(),
                last_session: None,
            },
            curriculum: Vec::new(),
            version: 0
        }
    }

    // Goal Discovery Phase
    pub async fn discover_goal(&mut self, api_key: &str, initial_topic: &str) -> Result<Goal, Box<dyn Error>> {
        log!("Starting goal discovery for topic: {}", initial_topic);
        
        let mut messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a learning assistant helping to define specific, actionable learning goals.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: format!("I want to learn about: {}", initial_topic),
            },
        ];
        log!("Created initial messages for goal discovery");

        let mut goal = Goal {
            id: Uuid::new_v4(),
            description: initial_topic.to_string(),
            criteria: Vec::new(),
            tags: Vec::new(),
            status: GoalStatus::Discovery,
            created_at: Utc::now(),
        };

        // Iterative refinement loop
        for _ in 0..3 {  // Maximum 3 refinement steps
            let response = self.generate_goal_refinement(&api_key, &messages).await?;
            messages.push(response.clone());
            
            // Check if goal is well-defined
            if self.evaluate_goal_completeness(&api_key, &messages).await? {
                goal.status = GoalStatus::Active;
                break;
            }
        }

        self.goals.push(goal.clone());
        Ok(goal)
    }

    // Flashcard Generation
    fn sync_goal_tags(&mut self, goal_id: Uuid) {
        let mut all_tags = std::collections::HashSet::new();
        
        // Collect all unique tags from cards associated with this goal
        for card in &self.cards {
            if card.goal_id == goal_id {
                all_tags.extend(card.tags.iter().cloned());
            }
        }
        
        // Update the goal's tags if we found it
        if let Some(goal) = self.goals.iter_mut().find(|g| g.id == goal_id) {
            goal.tags = all_tags.into_iter().collect();
        }
    }

    pub async fn generate_cards_for_goal(&mut self, api_key: &str, goal_id: Uuid) -> Result<Vec<Card>, Box<dyn Error>> {
        let goal = self.goals.iter()
            .find(|g| g.id == goal_id)
            .ok_or("Goal not found")?;

        // Get all relevant curriculum modules
        let curriculum_context = self.curriculum.iter()
            .filter(|module| module.topics.iter().any(|topic| 
                topic.cards.iter().any(|card_id| 
                    self.cards.iter().any(|c| c.id == *card_id && c.goal_id == goal_id)
                )
            ))
            .collect::<Vec<_>>();

        // Gather learning points from recent discussions
        let recent_learning_points = self.discussions.iter()
            .filter(|d| {
                let card = self.cards.iter().find(|c| c.id == d.card_id);
                card.map_or(false, |c| c.goal_id == goal_id)
            })
            .flat_map(|d| d.learning_points.clone())
            .collect::<Vec<_>>();

        // Get struggling areas (cards with low success rates)
        let struggling_cards = self.cards.iter()
            .filter(|c| c.goal_id == goal_id && c.success_rate < 0.7)
            .collect::<Vec<_>>();

        // Build comprehensive prompt
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: format!(
                    r#"You are an expert educational content creator designing flashcards.

For each card, follow this structure:
1. Question: Clear, specific question that tests one concept
2. Answer: Concise, complete answer with key points
3. Context: Supporting information without giving away the answer
4. Difficulty: Assign 1-5 based on concept complexity and prerequisites
5. Tags: Relevant topic tags for categorization

Consider:
- Student's weak areas: {}
- Current mastery level: {}
- Curriculum alignment: {}

Generate cards that:
- Build progressively on mastered concepts
- Address identified knowledge gaps
- Reinforce curriculum objectives
- Use varied question types (recall, application, analysis)
- Ensure questions are unambiguous and testable
- Include practical applications where relevant

Return exactly 5 cards in this JSON format:
{{
    "cards": [
        {{
            "question": "string",
            "answer": "string",
            "context": "string",
            "difficulty": number,
            "tags": ["string"]
        }}
    ]
}}"#,
                    struggling_cards.iter()
                        .map(|c| format!("{} ({}% success)", c.question, (c.success_rate * 100.0) as i32))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    recent_learning_points.iter()
                        .map(|lp| format!("{} (mastery: {})", lp.content, lp.mastery_level))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    // Calculate average mastery across topics
                    curriculum_context.iter()
                        .map(|m| format!("Module: {} - {}", m.title, m.description))
                        .collect::<Vec<_>>()
                        .join("\n")
                ),
            },
            ChatMessage {
                role: "user".to_string(),
                content: format!("Generate flashcards for this learning goal: {}", goal.description),
            },
        ];

        let cards = self.generate_structured_cards(&api_key, &messages).await?;
        self.cards.extend(cards.clone());
        
        // Sync tags after adding new cards
        self.sync_goal_tags(goal_id);
        
        Ok(cards)
    }

    // Interactive Learning
    pub async fn evaluate_response(&mut self, api_key: &str, card_id: Uuid, user_response: &str) -> Result<Discussion, Box<dyn Error>> {
        log!("Starting response evaluation for card {}", card_id);
        
        let card = match self.cards.iter()
            .find(|c| c.id == card_id) {
                Some(c) => c,
                None => {
                    log!("ERROR: Card {} not found", card_id);
                    return Err("Card not found".into());
                }
            };

        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: format!(
                    r#"You are an expert educational evaluator providing detailed feedback on student responses.

Evaluate the response considering:
1. Accuracy: Correctness of key concepts and details
2. Completeness: Coverage of all required elements
3. Understanding: Depth of conceptual grasp
4. Application: Ability to apply concepts correctly

Provide feedback in this JSON format:
{{
    "score": number (0.0-1.0),
    "critique": {{
        "strengths": ["string"],
        "areas_for_improvement": ["string"],
        "misconceptions": ["string"]
    }},
    "learning_points": [
        {{
            "content": "string - specific point of learning",
            "mastery": number (0.0-1.0),
            "improvement_suggestion": "string"
        }}
    ],
    "next_steps": ["string - specific actions to improve understanding"]
}}

Be specific, constructive, and actionable in your feedback.
Focus on guiding improvement rather than just pointing out errors."#
                ),
            },
            ChatMessage {
                role: "user".to_string(),
                content: format!(
                    r#"Question: {}
Correct Answer: {}
User Response: {}
Context: {}
Previous Performance: {} reviews, {}% success rate"#,
                    card.question,
                    card.answer,
                    user_response,
                    card.context,
                    card.review_count,
                    (card.success_rate * 100.0) as i32
                ),
            },
        ];

        log!("Sending evaluation request to OpenAI API");
        let response = self.generate_chat_completion(
            api_key,
            messages,
            "gpt-4",
            Some(0.7),
            Some(500),
        ).await?;

        let eval: serde_json::Value = serde_json::from_str(&response.choices[0].message.content)?;
        
        let discussion = Discussion {
            id: Uuid::new_v4(),
            card_id,
            user_response: user_response.to_string(),
            correctness_score: eval["score"].as_f64().unwrap_or(0.0) as f32,
            critique: eval["critique"].as_str().unwrap_or("").to_string(),
            learning_points: eval["learning_points"].as_array()
                .unwrap_or(&Vec::new())
                .iter()
                .map(|point| LearningPoint {
                    content: point.as_str().unwrap_or("").to_string(),
                    timestamp: Utc::now(),
                    mastery_level: 0.0,
                })
                .collect(),
            timestamp: Utc::now(),
        };

        log!("Successfully generated evaluation with score: {}", discussion.correctness_score);
        self.discussions.push(discussion.clone());
        self.update_statistics(card_id, &discussion);
        Ok(discussion)
    }

    // Progress Tracking
    fn update_statistics(&mut self, card_id: Uuid, discussion: &Discussion) {
        if let Some(card) = self.cards.iter_mut().find(|c| c.id == card_id) {
            card.review_count += 1;
            card.last_reviewed = Some(Utc::now());
            card.success_rate = ((card.success_rate * (card.review_count - 1) as f32) + discussion.correctness_score) 
                / card.review_count as f32;
            
            // Update tag performance
            for tag in &card.tags {
                if let Some(tag_perf) = self.progress.tag_performance.iter_mut()
                    .find(|t| t.tag == *tag) {
                    tag_perf.total_attempts += 1;
                    if discussion.correctness_score >= 0.8 {
                        tag_perf.success_count += 1;
                    } else {
                        tag_perf.failure_count += 1;
                    }
                    tag_perf.average_score = (tag_perf.average_score * (tag_perf.total_attempts - 1) as f32 
                        + discussion.correctness_score) / tag_perf.total_attempts as f32;
                }
            }
        }
    }


    async fn generate_chat_completion(
        &self,
        api_key: &str,
        messages: Vec<ChatMessage>,
        model: &str,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> Result<ChatCompletionResponse, Box<dyn Error>> {
        log!("Starting chat completion request to OpenAI API");
        log!("Using model: {}", model);
        
        let client = Client::new();
        let request = ChatCompletionRequest {
            model: model.to_string(),
            messages: messages.clone(),
            temperature,
            max_tokens,
            n: Some(1),
        };

        // Log the complete request payload
        log!("Request payload: {}", serde_json::to_string_pretty(&request).unwrap_or_default());

        log!("Sending request to OpenAI API...");
        let response = match client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&request)
            .send()
            .await {
                Ok(resp) => resp,
                Err(e) => {
                    log!("ERROR: Failed to send request to OpenAI API: {}", e);
                    return Err(Box::new(e));
                }
            };

        log!("Received response with status: {}", response.status());

        if response.status().is_success() {
            // Clone the response body for logging
            let response_text = response.text().await?;
            log!("Raw API response: {}", response_text);
            
            match serde_json::from_str(&response_text) {
                Ok(result) => {
                    log!("Successfully parsed API response");
                    Ok(result)
                },
                Err(e) => {
                    log!("ERROR: Failed to parse API response: {}", e);
                    log!("Failed response content: {}", response_text);
                    Err(Box::new(e))
                }
            }
        } else {
            let status = response.status();  // Store status before consuming response
            let error_text = match response.text().await {
                Ok(text) => text,
                Err(e) => format!("Could not read error response: {}", e),
            };
            log!("ERROR: API request failed with status {}: {}", 
                status, error_text);  // Use stored status
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("API request failed ({}): {}", status, error_text),  // Use stored status
            )))
        }
    }

    async fn generate_goal_refinement(
        &self,
        api_key: &str,
        messages: &[ChatMessage],
    ) -> Result<ChatMessage, Box<dyn Error>> {
        let prompt = format!(
            r#"You are helping to refine a learning goal.
Based on the user's response, suggest 2-3 specific, measurable criteria for success.

Please analyze the previous messages and return ONLY new criteria, one per line.
Each criterion should be:
1. Specific and measurable
2. Achievable within a reasonable timeframe
3. Relevant to the learning goal
4. Clear enough to create flashcards from

Format your response as simple text with one criterion per line."#
        );

        let mut refinement_messages = messages.to_vec();
        refinement_messages.push(ChatMessage {
            role: "system".to_string(),
            content: prompt,
        });

        let response = self.generate_chat_completion(
            api_key,
            refinement_messages,
            "gpt-4",
            Some(0.7),
            Some(500),
        ).await?;

        Ok(response.choices[0].message.clone())
    }

    async fn evaluate_goal_completeness(
        &self,
        api_key: &str,
        messages: &[ChatMessage],
    ) -> Result<bool, Box<dyn Error>> {
        let prompt = "Evaluate if the learning goal is specific, measurable, and actionable enough. \
                     Consider: 1) Is it clear what success looks like? 2) Can progress be measured? \
                     3) Is it focused enough to create specific flashcards? \
                     Respond with only 'true' if the goal is ready, or 'false' if it needs more refinement.";

        let mut eval_messages = messages.to_vec();
        eval_messages.push(ChatMessage {
            role: "system".to_string(),
            content: prompt.to_string(),
        });

        let response = self.generate_chat_completion(
            api_key,
            eval_messages,
            "gpt-4o-mini",
            Some(0.3),
            Some(50),
        ).await?;

        Ok(response.choices[0].message.content.trim().to_lowercase() == "true")
    }

    async fn generate_structured_cards(
        &self,
        api_key: &str,
        messages: &[ChatMessage],
    ) -> Result<Vec<Card>, Box<dyn Error>> {
        let prompt = r#"Create 5 flashcards for this learning goal. Not that the context will be shown to the user so it shouldn't give away the answer.
Return a JSON array where each card has the following properties:
{
    "question": "string - The question to ask",
    "answer": "string - The complete correct answer",
    "context": "string - Additional explanation or context for the topic",
    "difficulty": "number (1-5) - The difficulty level of the card"
}
Format your entire response as a valid JSON array of these objects."#;

        let mut card_messages = messages.to_vec();
        card_messages.push(ChatMessage {
            role: "system".to_string(),
            content: prompt.to_string(),
        });

        let response = self.generate_chat_completion(
            api_key,
            card_messages,
            "gpt-4o-mini",
            Some(0.7),
            Some(1000),
        ).await?;

        // Clean up the response content by removing markdown code block markers
        let content = response.choices[0].message.content.replace("```json", "")
            .replace("```", "")
            .trim()
            .to_string();

        log!("Cleaned JSON content: {}", content);

        let cards_json: Vec<serde_json::Value> = serde_json::from_str(&content)?;
    
        let cards = cards_json.into_iter().map(|card_json| {
            Card {
                id: Uuid::new_v4(),
                goal_id: self.goals.last().unwrap().id, // Link to the current goal
                question: card_json["question"].as_str().unwrap_or("").to_string(),
                answer: card_json["answer"].as_str().unwrap_or("").to_string(),
                context: card_json["context"].as_str().unwrap_or("").to_string(),
                difficulty: card_json["difficulty"].as_u64().unwrap_or(3) as u8,
                tags: self.goals.last().unwrap().tags.clone(),
                created_at: Utc::now(),
                review_count: 0,
                success_rate: 0.0,
                spaced_rep: SpacedRepetitionInfo {
                    last_reviewed: Utc::now(),
                    next_review: Utc::now(),
                    interval: 1,
                    ease_factor: 2.5,
                    consecutive_correct: 0,
                },
                prerequisites: Vec::new(),
                last_reviewed: None
            }
        }).collect();

        Ok(cards)
    }

    async fn generate_evaluation(
        &self,
        api_key: &str,
        messages: &[ChatMessage],
    ) -> Result<Discussion, Box<dyn Error>> {
        let prompt = r#"Evaluate the user's response to this flashcard. 
Return a JSON object with exactly these properties:
{
    "score": "number (0.0-1.0) - The correctness score",
    "critique": "string - Detailed critique of the response",
    "learning_points": "array of strings - Key points for improvement"
}
Format your entire response as a valid JSON object with these exact properties."#;

        log!("Generating evaluation with prompt: {}", prompt);
        
        let mut eval_messages = messages.to_vec();
        eval_messages.push(ChatMessage {
            role: "system".to_string(),
            content: prompt.to_string(),
        });

        log!("Full evaluation messages: {:?}", eval_messages);

        let response = self.generate_chat_completion(
            api_key,
            eval_messages,
            "gpt-4o-mini",
            Some(0.7),
            Some(500),
        ).await?;

        log!("Parsing evaluation response: {:?}", response);

        // Specify the type as Value for serde_json parsing
        let eval_json: serde_json::Value = serde_json::from_str(&response.choices[0].message.content)
            .map_err(|e| {
                log!("ERROR: Failed to parse evaluation JSON: {}", e);
                log!("Raw response content: {}", response.choices[0].message.content);
                e
            })?;

        // Wrap the Discussion in Ok() since we're returning a Result
        Ok(Discussion {
            id: Uuid::new_v4(),
            card_id: Uuid::nil(), // This should be set by the caller
            user_response: messages.last().unwrap().content.clone(),
            correctness_score: eval_json["score"].as_f64().unwrap_or(0.0) as f32,
            critique: eval_json["critique"].as_str().unwrap_or("").to_string(),
            learning_points: eval_json["learning_points"]
                .as_array()
                .unwrap_or(&Vec::new())
                .iter()
                .map(|point| point.as_str().unwrap_or("").to_string())
                .collect(),
            timestamp: Utc::now(),
        })
    }

    

    // Persistence Methods
    pub fn save(&self, file_path: &str) -> Result<(), Box<dyn Error>> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(file_path, json)?;
        Ok(())
    }

    pub fn load(file_path: &str) -> Result<Self, Box<dyn Error>> {
        let data = fs::read_to_string(file_path)?;
        let system: LearningSystem = serde_json::from_str(&data)?;
        Ok(system)
    }

}

async fn show_login() -> Html<String> {
    Html(r#"
        <!DOCTYPE html>
        <html>
        <head>
            <title>Learning System Login</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .login-form { max-width: 300px; margin: 0 auto; }
                input[type="password"] { width: 100%; padding: 8px; margin: 10px 0; }
                button { width: 100%; padding: 10px; background: #007bff; color: white; border: none; }
            </style>
        </head>
        <body>
            <div class="login-form">
                <h2>Login</h2>
                <form action="/login" method="POST">
                    <input type="password" name="password" placeholder="Password" required>
                    <button type="submit">Login</button>
                </form>
            </div>
        </body>
        </html>
    "#.to_string())
}

async fn handle_login(
    State(state): State<Arc<Mutex<AppState>>>,
    Form(form): Form<LoginForm>,
) -> Result<impl IntoResponse, AppError> {
    let mut state = state.lock().map_err(|_| AppError::SystemError("Lock error".to_string()))?;
    let ip = "127.0.0.1".to_string(); // In production, extract real IP
    
    // Check if IP is blocked
    if let Some(attempt) = state.login_attempts.get(&ip) {
        if attempt.attempts >= 3 {
            let elapsed = SystemTime::now()
                .duration_since(attempt.last_attempt)
                .unwrap_or(Duration::from_secs(0));
            
            if elapsed < Duration::from_secs(1800) { // 30 minutes
                return Ok(Html(
                    "Too many failed attempts. Please try again later.".to_string()
                ));
            }
        }
    }

    // Verify password
    if verify(&form.password, &state.password_hash)
        .map_err(|_| AppError::SystemError("Password verification failed".to_string()))? 
    {
        state.login_attempts.remove(&ip);
        Ok(Html(r#"<script>window.location.href = '/dashboard';</script>"#.to_string()))
    } else {
        // Record failed attempt
        let attempt = state.login_attempts
            .entry(ip)
            .or_insert(LoginAttempt { attempts: 0, last_attempt: SystemTime::now() });
        
        attempt.attempts += 1;
        attempt.last_attempt = SystemTime::now();

        Ok(Html(
            "Invalid password. Please try again.".to_string()
        ))
    }
}

#[axum::debug_handler]
async fn show_practice_session(
    State(state): State<Arc<Mutex<AppState>>>,
    Path(goal_id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    let state = state.lock().map_err(|_| AppError::SystemError("Lock error".to_string()))?;
    
    // Generate a 30-minute practice session
    let practice_cards = state.learning_system.generate_practice_session(goal_id, 30);
    
    Ok(Html(format!(r#"
        <!DOCTYPE html>
        <html>
        <head>
            <title>Practice Session</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .card {{ 
                    background: #f8f9fa; 
                    padding: 20px; 
                    margin: 20px 0; 
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .progress-bar {{
                    background: #e9ecef;
                    height: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                }}
                .progress {{
                    background: #007bff;
                    height: 100%;
                    border-radius: 10px;
                    width: 0%;
                    transition: width 0.5s;
                }}
                .answer-form {{ margin: 15px 0; }}
                .answer-input {{ 
                    width: 100%; 
                    min-height: 100px; 
                    margin: 10px 0; 
                    padding: 8px;
                }}
                .submit-btn {{ 
                    background: #007bff; 
                    color: white; 
                    border: none;
                    padding: 10px 20px;
                    cursor: pointer;
                    border-radius: 4px;
                }}
            </style>
        </head>
        <body>
            <h1>Practice Session</h1>
            <div class="progress-bar">
                <div class="progress" id="progress"></div>
            </div>
            
            <div class="cards">
                {}
            </div>

            <script>
                // Update progress as cards are completed
                let totalCards = {};
                let completedCards = 0;
                
                function updateProgress() {{
                    completedCards++;
                    let progress = (completedCards / totalCards) * 100;
                    document.getElementById('progress').style.width = progress + '%';
                }}
            </script>
        </body>
        </html>
    "#,
        practice_cards.iter().map(|card| format!(r#"
            <div class="card">
                <h3>{}</h3>
                <p><strong>Context:</strong> {}</p>
                <form class="answer-form" action="/practice/{}/submit/{}" method="POST" 
                      onsubmit="updateProgress()">
                    <textarea 
                        class="answer-input" 
                        name="user_answer" 
                        placeholder="Type your answer here..."
                        required
                    ></textarea>
                    <button type="submit" class="submit-btn">Submit Answer</button>
                </form>
            </div>
        "#,
            card.question,
            card.context,
            goal_id,
            card.id
        )).collect::<Vec<_>>().join("\n"),
        practice_cards.len()
    )))
}

#[axum::debug_handler]
async fn handle_practice_submission(
    State(state): State<Arc<Mutex<AppState>>>,
    Path((goal_id, card_id)): Path<(Uuid, Uuid)>,
    Form(form): Form<AnswerSubmission>,
) -> Result<impl IntoResponse, AppError> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| AppError::SystemError("API key not found".to_string()))?;

    // Get what we need from the locked state
    let (card_clone, performance_history, mut learning_system) = {
        let state = state.lock().map_err(|_| AppError::SystemError("Lock error".to_string()))?;
        let card = state.learning_system.cards.iter()
            .find(|c| c.id == card_id)
            .ok_or_else(|| AppError::NotFound("Card not found".to_string()))?
            .clone();
        let history: Vec<f32> = state.learning_system.discussions.iter()
            .filter(|d| d.card_id == card_id)
            .map(|d| d.correctness_score)
            .collect();
        (card, history, state.learning_system.clone())
    };

    // Adjust card difficulty based on performance
    if let Err(e) = learning_system.adjust_card_difficulty(card_id, &performance_history) {
        log!("Error adjusting card difficulty: {}", e);
    }

    // Evaluate the response using cloned data
    let discussion = learning_system.evaluate_response(&api_key, card_id, &form.user_answer)
        .await
        .map_err(|e| AppError::SystemError(e.to_string()))?;

    // Calculate new spaced repetition info
    let new_spaced_rep = learning_system.calculate_next_review(&card_clone, discussion.correctness_score);

    // Update the original state
    {
        let mut state = state.lock().map_err(|_| AppError::SystemError("Lock error".to_string()))?;
        if let Some(card) = state.learning_system.cards.iter_mut().find(|c| c.id == card_id) {
            card.spaced_rep = new_spaced_rep;
        }
        state.learning_system = learning_system;
        if let Err(e) = state.learning_system.save("learning_system.json") {
            log!("Error saving state: {}", e);
        }
    }

    Ok(Html(format!(r#"
        <!DOCTYPE html>
        <html>
        <head>
            <title>Practice Feedback</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .feedback {{ 
                    background: #f8f9fa; 
                    padding: 20px; 
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                .score {{ 
                    font-size: 1.2em; 
                    font-weight: bold; 
                    color: {}; 
                }}
                .next-btn {{
                    display: inline-block;
                    padding: 10px 20px;
                    background: #007bff;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="feedback">
                <div class="score">Score: {:.1}%</div>
                <h3>Feedback:</h3>
                <p>{}</p>
                <h3>Key Points:</h3>
                <ul>
                    {}
                </ul>
            </div>
            <a href="/practice/{}" class="next-btn">Next Card</a>
        </body>
        </html>
    "#,
        if discussion.correctness_score >= 0.8 { "#28a745" } else { "#dc3545" },
        discussion.correctness_score * 100.0,
        discussion.critique,
        discussion.learning_points.iter()
            .map(|point| format!("<li>{}</li>", point))
            .collect::<Vec<_>>()
            .join("\n"),
        goal_id
    )))
}

async fn show_knowledge_graph(
    State(state): State<Arc<Mutex<AppState>>>,
    Path(goal_id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    let state = state.lock().map_err(|_| AppError::SystemError("Lock error".to_string()))?;
    let graph_data = state.learning_system.generate_knowledge_graph(goal_id);
    
    Ok(Html(format!(r#"
        <!DOCTYPE html>
        <html>
        <head>
            <title>Knowledge Graph</title>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                .node {{
                    fill: #66b2ff;
                    stroke: #fff;
                    stroke-width: 2px;
                }}
                .node.mastered {{
                    fill: #28a745;
                }}
                .node.weak {{
                    fill: #dc3545;
                }}
                .link {{
                    stroke: #999;
                    stroke-opacity: 0.6;
                    stroke-width: 1px;
                }}
                .node-label {{
                    font-family: Arial;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div id="graph"></div>
            <script>
                const data = {};
                
                const width = window.innerWidth;
                const height = window.innerHeight;

                const svg = d3.select('#graph')
                    .append('svg')
                    .attr('width', width)
                    .attr('height', height);

                const simulation = d3.forceSimulation(data.nodes)
                    .force('link', d3.forceLink(data.edges)
                        .id(d => d.id)
                        .distance(100))
                    .force('charge', d3.forceManyBody().strength(-200))
                    .force('center', d3.forceCenter(width / 2, height / 2));

                const link = svg.append('g')
                    .selectAll('line')
                    .data(data.edges)
                    .enter().append('line')
                    .attr('class', 'link');

                const node = svg.append('g')
                    .selectAll('circle')
                    .data(data.nodes)
                    .enter().append('circle')
                    .attr('class', d => `node ${{d.mastery > 0.8 ? 'mastered' : d.mastery < 0.4 ? 'weak' : ''}}`)
                    .attr('r', d => 5 + d.level * 2)
                    .call(d3.drag()
                        .on('start', dragstarted)
                        .on('drag', dragged)
                        .on('end', dragended));

                const label = svg.append('g')
                    .selectAll('text')
                    .data(data.nodes)
                    .enter().append('text')
                    .attr('class', 'node-label')
                    .text(d => d.label);

                simulation.on('tick', () => {{
                    link
                        .attr('x1', d => d.source.x)
                        .attr('y1', d => d.source.y)
                        .attr('x2', d => d.target.x)
                        .attr('y2', d => d.target.y);

                    node
                        .attr('cx', d => d.x)
                        .attr('cy', d => d.y);

                    label
                        .attr('x', d => d.x + 10)
                        .attr('y', d => d.y + 3);
                }});

                function dragstarted(event, d) {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }}

                function dragged(event, d) {{
                    d.fx = event.x;
                    d.fy = event.y;
                }}

                function dragended(event, d) {{
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }}
            </script>
        </body>
        </html>
    "#, serde_json::to_string(&graph_data)?)))
}

async fn handle_answer_submission(
    State(state): State<Arc<Mutex<AppState>>>,
    Path((goal_id, card_id)): Path<(Uuid, Uuid)>,
    Form(form): Form<AnswerSubmission>,
) -> Result<impl IntoResponse, AppError> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| AppError::SystemError("API key not found".to_string()))?;

    // Clone the learning system
    let mut learning_system = {
        let state = state.lock().map_err(|_| AppError::SystemError("Lock error".to_string()))?;
        state.learning_system.clone()
    };

    // Evaluate response using the cloned system
    let discussion = learning_system.evaluate_response(&api_key, card_id, &form.user_answer)
        .await
        .map_err(|e| AppError::SystemError(e.to_string()))?;

    // Update the original state with the results
    {
        let mut state = state.lock().map_err(|_| AppError::SystemError("Lock error".to_string()))?;
        state.learning_system = learning_system;
        if let Err(e) = state.learning_system.save("learning_system.json") {
            eprintln!("Error saving state: {}", e);
        }
    }

            Ok(Html(format!(r#"
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Answer Feedback</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        .feedback-container {{ max-width: 800px; margin: 0 auto; }}
                        .score {{ 
                            font-size: 1.2em; 
                            font-weight: bold; 
                            color: {}; 
                            margin: 20px 0;
                        }}
                        .critique {{ 
                            background: #f8f9fa; 
                            padding: 20px; 
                            border-radius: 8px; 
                            margin: 20px 0;
                        }}
                        .learning-points {{ margin: 20px 0; }}
                        .learning-point {{ 
                            padding: 10px;
                            margin: 5px 0;
                            background: #e9ecef;
                            border-radius: 4px;
                        }}
                        .nav-buttons {{ margin-top: 30px; }}
                        .nav-buttons a {{
                            display: inline-block;
                            padding: 10px 20px;
                            background: #007bff;
                            color: white;
                            text-decoration: none;
                            border-radius: 4px;
                            margin-right: 10px;
                        }}
                    </style>
                </head>
                <body>
                    <div class="feedback-container">
                        <h1>Answer Feedback</h1>
                        
                        <div class="score">
                            Score: {:.1}%
                        </div>

                        <div class="critique">
                            <h3>Critique:</h3>
                            <p>{}</p>
                        </div>

                        <div class="learning-points">
                            <h3>Key Learning Points:</h3>
                            {}
                        </div>

                        <div class="nav-buttons">
                            <a href="/study/{}">Continue Studying</a>
                            <a href="/dashboard">Back to Dashboard</a>
                        </div>
                    </div>
                </body>
                </html>
            "#,
                if discussion.correctness_score >= 0.8 { "#28a745" } else { "#dc3545" },
                discussion.correctness_score * 100.0,
                discussion.critique,
                discussion.learning_points.iter()
                    .map(|point| format!("<div class=\"learning-point\">{}</div>", point.content))
                    .collect::<Vec<_>>()
                    .join("\n"),
                goal_id
            )))
        }


async fn show_dashboard(
    State(state): State<Arc<Mutex<AppState>>>,
) -> Html<String> {
    let state = state.lock().unwrap();
    
    Html(format!(r#"
        <!DOCTYPE html>
        <html>
        <head>
            <title>Learning Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .card {{ 
                    background: #ffffff; 
                    padding: 20px; 
                    margin: 10px 0; 
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    border: 1px solid #dee2e6;
                }}
                .card h3 {{ margin-top: 0; }}
                .action-buttons {{
                    display: flex;
                    gap: 10px;
                    margin-top: 15px;
                }}
                .action-button {{
                    flex: 1;
                    text-align: center;
                    padding: 8px 16px;
                    background: #007bff;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                    font-size: 0.9em;
                }}
                .action-button:hover {{
                    background: #0056b3;
                }}
                .status-badge {{
                    display: inline-block;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 0.8em;
                    margin-left: 10px;
                }}
                .status-active {{ background: #28a745; color: white; }}
                .status-discovery {{ background: #ffc107; color: black; }}
                .status-completed {{ background: #6c757d; color: white; }}
                .action-button {{
                    display: inline-block;
                    padding: 8px 16px;
                    background: #007bff;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                    margin-top: 10px;
                }}
                .new-goal-button {{
                    display: inline-block;
                    padding: 12px 24px;
                    background: #28a745;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                    margin-top: 20px;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <h1>Learning Dashboard</h1>

            <div class="weak-topics">
                <h2>Topics Needing Review</h2>
                <ul>
                    {}
                </ul>
            </div>

            <div class="goals">
                <h2>Your Learning Goals</h2>
                {}
            </div>

            <a href="/goals/new" class="new-goal-button">
                + Create New Learning Goal
            </a>
        </body>
        </html>
    "#,
    state.learning_system.identify_weak_topics()
        .iter()
        .map(|topic| format!("<li>{}</li>", topic))
        .collect::<Vec<_>>()
        .join("\n"),
    state.learning_system.goals
        .iter()
        .map(|g| {
            let status_class = match g.status {
                GoalStatus::Active => "status-active",
                GoalStatus::Discovery => "status-discovery",
                GoalStatus::Completed => "status-completed",
                GoalStatus::Archived => "status-completed",
            };
            
            let status_text = format!("{:?}", g.status);
            
            format!(r#"
                <div class="card">
                    <h3>
                        {}
                        <span class="status-badge {}">{}</span>
                    </h3>
                    <p><strong>Tags:</strong> {}</p>
                    <div>
                        <strong>Success Criteria:</strong>
                        <ul>
                            {}
                        </ul>
                    </div>
                    <div class="action-buttons">
                        <a href="/study/{}" class="action-button">
                            Continue Studying
                        </a>
                        <a href="/practice/{}" class="action-button">
                            Start Practice Session
                        </a>
                        <a href="/knowledge-graph/{}" class="action-button">
                            View Knowledge Graph
                        </a>
                    </div>
                </div>
            "#,
            g.description,
            status_class,
            status_text,
            g.tags.join(", "),
            g.criteria.iter()
                .map(|c| format!("<li>{}</li>", c))
                .collect::<Vec<_>>()
                .join("\n"),
            g.id,
            g.id,
            g.id
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
    ))
}

async fn show_goal_form() -> Html<String> {
    Html(r#"
        <!DOCTYPE html>
        <html>
        <head>
            <title>Create New Learning Goal</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .form-container { max-width: 600px; margin: 0 auto; }
                textarea, input { width: 100%; padding: 8px; margin: 10px 0; }
                button { padding: 10px; background: #007bff; color: white; border: none; cursor: pointer; }
            </style>
        </head>
        <body>
            <div class="form-container">
                <h2>Create New Learning Goal</h2>
                <form action="/goals/new" method="POST">
                    <div>
                        <label>What do you want to learn about?</label>
                        <textarea name="topic" rows="3" required 
                            placeholder="Describe what you want to learn. Be as specific as possible."></textarea>
                    </div>
                    <button type="submit">Generate Learning Plan</button>
                </form>
            </div>
        </body>
        </html>
    "#.to_string())
}

#[derive(Deserialize)]
struct GoalForm {
    topic: String,
}

#[derive(Deserialize)]
struct AnswerSubmission {
    user_answer: String,
}

#[derive(Deserialize)]
struct RefinementResponse {
    response: String,
}

async fn show_goal_refinement(
    State(state): State<Arc<Mutex<AppState>>>,
    Path(goal_id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    let state = state.lock().map_err(|_| AppError::SystemError("Lock error".to_string()))?;
    
    let goal = state.learning_system.goals.iter()
        .find(|g| g.id == goal_id)
        .ok_or_else(|| AppError::NotFound("Goal not found".to_string()))?;

    Ok(Html(format!(r#"
        <!DOCTYPE html>
        <html>
        <head>
            <title>Refine Learning Goal</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .container {{ max-width: 800px; margin: 0 auto; }}
                .question-box {{ 
                    background: #f8f9fa; 
                    padding: 20px; 
                    border-radius: 8px; 
                    margin: 20px 0; 
                }}
                .answer-input {{ 
                    width: 100%; 
                    min-height: 100px; 
                    margin: 10px 0; 
                    padding: 8px; 
                }}
                .submit-btn {{ 
                    background: #007bff; 
                    color: white; 
                    padding: 10px 20px; 
                    border: none; 
                    border-radius: 4px; 
                    cursor: pointer; 
                }}
                .criteria-list {{ margin: 20px 0; }}
                .criteria-item {{ 
                    background: #e9ecef; 
                    padding: 10px; 
                    margin: 5px 0; 
                    border-radius: 4px; 
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Refine Your Learning Goal</h1>
                <p><strong>Current Goal:</strong> {}</p>
                
                <div class="criteria-list">
                    <h3>Current Criteria:</h3>
                    {}
                </div>

                <div class="question-box">
                    <form action="/goals/{}/refine" method="POST">
                        <h3>Please answer this question to help refine your goal:</h3>
                        <p>{}</p>
                        <textarea 
                            class="answer-input" 
                            name="response" 
                            required 
                            placeholder="Type your answer here..."
                        ></textarea>
                        <button type="submit" class="submit-btn">Submit Answer</button>
                    </form>
                </div>
            </div>
        </body>
        </html>
    "#,
        goal.description,
        goal.criteria.iter()
            .map(|c| format!("<div class=\"criteria-item\">{}</div>", c))
            .collect::<Vec<_>>()
            .join("\n"),
        goal_id,
        "What specific aspects of this topic are you most interested in learning about?"
    )))
}

#[axum::debug_handler]
async fn handle_goal_refinement(
    State(state): State<Arc<Mutex<AppState>>>,
    Path(goal_id): Path<Uuid>,
    Form(form): Form<RefinementResponse>,
) -> Result<impl IntoResponse, AppError> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| AppError::SystemError("API key not found".to_string()))?;

    // Clone the learning system first
    let mut learning_system = {
        let state = state.lock().map_err(|_| AppError::SystemError("Lock error".to_string()))?;
        state.learning_system.clone()
    };

    // Use the cloned system for all operations
    let goal = learning_system.goals.iter()
        .find(|g| g.id == goal_id)
        .ok_or_else(|| AppError::NotFound("Goal not found".to_string()))?;

    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: format!(
                r#"You are helping to refine a learning goal.
Based on the user's response, suggest 2-3 specific, measurable criteria for success.
Current goal description: {}
Current criteria: {}
Please return ONLY the new criteria, one per line."#,
                goal.description,
                goal.criteria.join("\n")
            ),
        },
        ChatMessage {
            role: "user".to_string(),
            content: form.response.clone(),
        },
    ];

    let completion_result = learning_system.generate_chat_completion(
        &api_key,
        messages,
        "gpt-4",
        Some(0.7),
        Some(500),
    ).await?;

    let should_generate_cards = {
        if let Some(goal) = learning_system.goals.iter_mut().find(|g| g.id == goal_id) {
            let new_criteria = completion_result.choices[0].message.content
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| line.trim().to_string())
                .collect::<Vec<_>>();

            goal.criteria.extend(new_criteria);
            
            if goal.criteria.len() >= 3 {
                goal.status = GoalStatus::Active;
                true
            } else {
                false
            }
        } else {
            return Err(AppError::NotFound("Goal not found".to_string()));
        }
    };

    if should_generate_cards {
        if let Err(e) = learning_system.generate_cards_for_goal(&api_key, goal_id).await {
            log!("ERROR: Failed to generate cards: {}", e);
        }
    }

    // Update the original state with our modified copy
    {
        let mut state = state.lock().map_err(|_| AppError::SystemError("Lock error".to_string()))?;
        state.learning_system = learning_system;
        if let Err(e) = state.learning_system.save("learning_system.json") {
            log!("ERROR: Failed to save learning system: {}", e);
        }
    }

    if should_generate_cards {
        Ok(Html(format!(
            r#"<script>window.location.href = '/study/{}';</script>"#,
            goal_id
        )))
    } else {
        Ok(Html(format!(
            r#"<script>window.location.href = '/goals/{}/refine';</script>"#,
            goal_id
        )))
    }
}

#[axum::debug_handler]
async fn handle_goal_creation(
    State(state): State<Arc<Mutex<AppState>>>,
    Form(form): Form<GoalForm>,
) -> Result<impl IntoResponse, AppError> {
    log!("Starting goal creation for topic: {}", form.topic);

    // Create initial goal
    let goal = Goal {
        id: Uuid::new_v4(),
        description: form.topic.clone(),
        criteria: Vec::new(),
        tags: Vec::new(),
        status: GoalStatus::Discovery,
        created_at: Utc::now(),
    };

    // Store the goal in the state
    {
        let mut state = state.lock().map_err(|_| AppError::SystemError("Lock error".to_string()))?;
        state.learning_system.goals.push(goal.clone());
        if let Err(e) = state.learning_system.save("learning_system.json") {
            log!("ERROR: Failed to save initial goal: {}", e);
        }
    }

    // Redirect to the refinement page
    Ok(Html(format!(
        r#"<script>window.location.href = '/goals/{}/refine';</script>"#,
        goal.id
    )))
}

#[axum::debug_handler]
async fn get_due_cards(
    State(state): State<Arc<Mutex<AppState>>>,
) -> Result<impl IntoResponse, AppError> {
    let state = state.lock().map_err(|_| AppError::SystemError("Lock error".to_string()))?;
    
    // Get due cards from learning system
    let due_cards = state.learning_system.get_due_cards();
    
    // Convert to JSON response
    Ok(Json(
        due_cards.into_iter()
            .map(|card| {
                json!({
                    "id": card.id,
                    "question": card.question,
                    "goal_id": card.goal_id,
                    "difficulty": card.difficulty,
                    "next_review": card.spaced_rep.next_review,
                    "tags": card.tags
                })
            })
            .collect::<Vec<_>>()
    ))
}

async fn show_study_page(
    State(state): State<Arc<Mutex<AppState>>>,
    Path(goal_id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    let state = state.lock().map_err(|_| AppError::SystemError("Lock error".to_string()))?;
    
    let goal = state.learning_system.goals.iter()
        .find(|g| g.id == goal_id)
        .ok_or_else(|| AppError::NotFound("Goal not found".to_string()))?;
    
    let cards = state.learning_system.cards.iter()
        .filter(|c| c.goal_id == goal_id)
        .collect::<Vec<_>>();

    Ok(Html(format!(r#"
        <!DOCTYPE html>
        <html>
        <head>
            <title>Study</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .card {{ 
                    background: #f5f5f5; 
                    padding: 20px; 
                    margin: 20px 0; 
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .answer-form {{ margin: 15px 0; }}
                .answer-input {{ 
                    width: 100%; 
                    min-height: 100px; 
                    margin: 10px 0; 
                    padding: 8px;
                }}
                .submit-btn {{ 
                    background: #007bff; 
                    color: white; 
                    border: none;
                    padding: 10px 20px;
                    cursor: pointer;
                    border-radius: 4px;
                }}
                .difficulty {{
                    color: #666;
                    font-size: 0.9em;
                    margin-top: 10px;
                }}
                .nav-bar {{
                    margin-bottom: 20px;
                }}
                .nav-bar a {{
                    color: #007bff;
                    text-decoration: none;
                }}
            </style>
        </head>
        <body>
            <div class="nav-bar">
                <a href="/dashboard">← Back to Dashboard</a>
                <a href="/practice/{}" class="practice-btn">Start 30-min Practice Session</a>
            </div>
            
            <h1>Welcome</h1>
            <p><strong>Goal:</strong> {}</p>
            
            <div class="cards">
                {}
            </div>
        </body>
        </html>
    "#,
        goal_id,
        goal.description,
        cards.iter().map(|card| format!(r#"
            <div class="card">
                <h3>{}</h3>
                <p><strong>Context:</strong> {}</p>
                <form class="answer-form" action="/study/{}/submit/{}" method="POST">
                    <textarea 
                        class="answer-input" 
                        name="user_answer" 
                        placeholder="Type your answer here..."
                        required
                    ></textarea>
                    <button type="submit" class="submit-btn">Submit Answer</button>
                </form>
                <div class="difficulty">
                    Difficulty: {}/5 | Reviews: {} | Success Rate: {:.1}%
                </div>
            </div>
        "#,
            card.question,
            card.context,
            goal_id,
            card.id,
            card.difficulty,
            card.review_count,
            card.success_rate * 100.0
        )).collect::<Vec<_>>().join("\n")
    )))
}

#[tokio::main]
async fn main() {
    log!("Starting Iterative Flashcard System");
    
    // Initialize system
    let _api_key = match std::env::var("OPENAI_API_KEY") {
        Ok(key) => {
            log!("Successfully loaded API key from environment");
            key
        },
        Err(e) => {
            log!("ERROR: Failed to load API key: {}", e);
            eprintln!("ERROR: OpenAI API key not found in environment");
            std::process::exit(1);
        }
    };
    
    log!("Initializing learning system");
    let system = match LearningSystem::load("learning_system.json") {
        Ok(loaded_system) => {
            log!("Successfully loaded existing learning system data");
            loaded_system
        },
        Err(e) => {
            log!("Could not load existing data ({}), starting with new system", e);
            LearningSystem::new()
        }
    };
    
    log!("Creating password hash");
    let password_hash = match hash("your_password_here", DEFAULT_COST) {
        Ok(hash) => hash,
        Err(e) => {
            eprintln!("Failed to create password hash: {}", e);
            std::process::exit(1);
        }
    };

    log!("Creating shared state");
    let state = Arc::new(Mutex::new(AppState {
        learning_system: system,
        login_attempts: HashMap::new(),
        password_hash,
    }));

    log!("Building router");
    let app = Router::new()
        .route("/", get(show_login))
        .route("/login", post(handle_login))
        .route("/dashboard", get(show_dashboard))
        .route("/goals/new", get(show_goal_form).post(handle_goal_creation))
        .route("/goals/:goal_id/refine", get(show_goal_refinement).post(handle_goal_refinement))
        .route("/study/:goal_id", get(show_study_page))
        .route("/study/:goal_id/submit/:card_id", post(handle_answer_submission))
        .route("/curriculum/:goal_id", post(update_curriculum))
        .route("/due-cards", get(get_due_cards))
        .route("/knowledge-graph/:goal_id", get(show_knowledge_graph))
        .route("/practice/:goal_id", get(show_practice_session))
        .route("/practice/:goal_id/submit/:card_id", post(handle_practice_submission))
        .with_state(state);

    log!("Starting server on http://localhost:3000");
    let listener = match tokio::net::TcpListener::bind("0.0.0.0:3000").await {
        Ok(listener) => {
            log!("Successfully bound to port 3000");
            listener
        },
        Err(e) => {
            eprintln!("Failed to bind to port 3000: {}", e);
            std::process::exit(1);
        }
    };

    log!("Server is ready to accept connections");
    if let Err(e) = axum::serve(listener, app.into_make_service()).await {
        eprintln!("Server error: {}", e);
        std::process::exit(1);
    }
}
