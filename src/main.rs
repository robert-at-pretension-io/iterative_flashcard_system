use axum::{
    response::{Response, IntoResponse, Html},
    routing::{get, post},
    extract::{State, Form, Path},
    Router,
    http::StatusCode,
};
use tokio::net::TcpListener;

#[derive(Debug)]
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Card {
    pub id: Uuid,
    pub goal_id: Uuid,
    pub question: String,
    pub answer: String,
    pub context: String,         // Required context/explanation
    pub difficulty: u8,          // 1-5 scale
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub last_reviewed: Option<DateTime<Utc>>,
    pub review_count: u32,
    pub success_rate: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Discussion {
    pub id: Uuid,
    pub card_id: Uuid,
    pub user_response: String,
    pub correctness_score: f32,  // 0-1 scale
    pub critique: String,
    pub learning_points: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TagPerformance {
    pub tag: String,
    pub total_attempts: u32,
    pub success_count: u32,
    pub failure_count: u32,
    pub average_score: f32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UserProgress {
    pub total_cards_reviewed: u64,
    pub total_study_sessions: u64,
    pub tag_performance: Vec<TagPerformance>,
    pub active_goals: Vec<Uuid>,
    pub completed_goals: Vec<Uuid>,
    pub last_session: Option<DateTime<Utc>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LearningSystem {
    pub goals: Vec<Goal>,
    pub cards: Vec<Card>,
    pub discussions: Vec<Discussion>,
    pub progress: UserProgress,
}

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


// ... [Main function remains the same] ...

// ---- Implementation ----

impl LearningSystem {
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
        }
    }

    // Goal Discovery Phase
    pub async fn discover_goal(&mut self, api_key: &str, initial_topic: &str) -> Result<Goal, Box<dyn Error>> {
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
    pub async fn generate_cards_for_goal(&mut self, api_key: &str, goal_id: Uuid) -> Result<Vec<Card>, Box<dyn Error>> {
        let goal = self.goals.iter()
            .find(|g| g.id == goal_id)
            .ok_or("Goal not found")?;

        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "Create a series of flashcards to help achieve this learning goal.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: format!("Goal: {}\nCriteria: {:?}", goal.description, goal.criteria),
            },
        ];

        let cards = self.generate_structured_cards(&api_key, &messages).await?;
        self.cards.extend(cards.clone());
        Ok(cards)
    }

    // Interactive Learning
    pub async fn evaluate_response(&mut self, api_key: &str, card_id: Uuid, user_response: &str) -> Result<Discussion, Box<dyn Error>> {
        let card = self.cards.iter()
            .find(|c| c.id == card_id)
            .ok_or("Card not found")?;

        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "Evaluate the user's response to this flashcard.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: format!(
                    "Question: {}\nCorrect Answer: {}\nUser Response: {}\nContext: {}",
                    card.question, card.answer, user_response, card.context
                ),
            },
        ];

        let discussion = self.generate_evaluation(&api_key, &messages).await?;
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
        let client = Client::new();
        let request = ChatCompletionRequest {
            model: model.to_string(),
            messages,
            temperature,
            max_tokens,
            n: Some(1),
        };

        let response = client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let result: ChatCompletionResponse = response.json().await?;
            Ok(result)
        } else {
            let error_message = response.text().await?;
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("API request failed: {}", error_message),
            )))
        }
    }

    async fn generate_goal_refinement(
        &self,
        api_key: &str,
        messages: &[ChatMessage],
    ) -> Result<ChatMessage, Box<dyn Error>> {
        let prompt = format!(
            "Based on the conversation so far, ask ONE specific question to help refine and clarify the learning goal. \
            Focus on making the goal more specific, measurable, and actionable.");

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
            Some(100),
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
            "gpt-4",
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
        let prompt = "Create 5 flashcards for this learning goal. \
                     Each card should have a clear question, comprehensive answer, relevant context, \
                     and appropriate difficulty level (1-5). Format as JSON array.";

        let mut card_messages = messages.to_vec();
        card_messages.push(ChatMessage {
            role: "system".to_string(),
            content: prompt.to_string(),
        });

        let response = self.generate_chat_completion(
            api_key,
            card_messages,
            "gpt-4",
            Some(0.7),
            Some(1000),
        ).await?;

        let cards_json: Vec<serde_json::Value> = serde_json::from_str(&response.choices[0].message.content)?;
        
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
                last_reviewed: None,
                review_count: 0,
                success_rate: 0.0,
            }
        }).collect();

        Ok(cards)
    }

    async fn generate_evaluation(
        &self,
        api_key: &str,
        messages: &[ChatMessage],
    ) -> Result<Discussion, Box<dyn Error>> {
        let prompt = "Evaluate the user's response to this flashcard. Provide: \
                     1) A correctness score (0-1), \
                     2) Detailed critique of the response, \
                     3) Key learning points for improvement. \
                     Format as JSON.";

        let mut eval_messages = messages.to_vec();
        eval_messages.push(ChatMessage {
            role: "system".to_string(),
            content: prompt.to_string(),
        });

        let response = self.generate_chat_completion(
            api_key,
            eval_messages,
            "gpt-4",
            Some(0.7),
            Some(500),
        ).await?;

        let eval_json: serde_json::Value = serde_json::from_str(&response.choices[0].message.content)?;
        
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

    // Helper methods for OpenAI API calls would go here
    // Including: generate_goal_refinement, evaluate_goal_completeness,
    // generate_structured_cards, generate_evaluation
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
async fn handle_answer_submission(
    State(state): State<Arc<Mutex<AppState>>>,
    Path((goal_id, card_id)): Path<(Uuid, Uuid)>,
    Form(form): Form<AnswerSubmission>,
) -> Result<impl IntoResponse, AppError> {
    let mut state = state.lock().map_err(|_| AppError::SystemError("Lock error".to_string()))?;
    
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| AppError::SystemError("API key not found".to_string()))?;

    let discussion = state.learning_system.evaluate_response(&api_key, card_id, &form.user_answer)
        .await
        .map_err(|e| AppError::SystemError(e.to_string()))?;

    if let Err(e) = state.learning_system.save("learning_system.json") {
        eprintln!("Error saving state: {}", e);
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
                    .map(|point| format!("<div class=\"learning-point\">{}</div>", point))
                    .collect::<Vec<_>>()
                    .join("\n"),
                goal_id
            )))
        }


async fn show_dashboard(
    State(state): State<Arc<Mutex<AppState>>>,
) -> Html<String> {
    let state = state.lock().unwrap();
    let progress = &state.learning_system.progress;
    
    Html(format!(r#"
        <!DOCTYPE html>
        <html>
        <head>
            <title>Learning Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .stats {{ margin: 20px 0; }}
                .card {{ background: #f5f5f5; padding: 20px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Learning Dashboard</h1>
            <div class="stats">
                <h2>Progress Statistics</h2>
                <p>Total Cards Reviewed: {}</p>
                <p>Total Study Sessions: {}</p>
            </div>
            <div class="goals">
                <h2>Active Goals</h2>
                {}
            </div>
            <div class="actions" style="margin-top: 20px;">
                <h2>Actions</h2>
                <a href="/goals/new" style="display: inline-block; padding: 10px; background: #007bff; color: white; text-decoration: none; border-radius: 4px;">
                    Create New Learning Goal
                </a>
            </div>
        </body>
        </html>
    "#, 
    progress.total_cards_reviewed,
    progress.total_study_sessions,
    state.learning_system.goals
        .iter()
        .map(|g| format!(
            r#"<div class="card"><h3>{}</h3><p>Status: {:?}</p></div>"#,
            g.description,
            g.status
        ))
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

#[axum::debug_handler]
async fn handle_goal_creation(
    State(state): State<Arc<Mutex<AppState>>>,
    Form(form): Form<GoalForm>,
) -> Result<impl IntoResponse, AppError> {
    let mut state = state.lock().map_err(|_| AppError::SystemError("Lock error".to_string()))?;
    
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| AppError::SystemError("API key not found".to_string()))?;

    let goal = state.learning_system.discover_goal(&api_key, &form.topic)
        .await
        .map_err(|e| AppError::SystemError(e.to_string()))?;

    // Generate initial flashcards
    if let Err(e) = state.learning_system.generate_cards_for_goal(&api_key, goal.id).await {
        eprintln!("Error generating cards: {}", e);
    }
    
    Ok(Html(format!(
        r#"<script>window.location.href = '/study/{}';</script>"#,
        goal.id
    )))
}

#[axum::debug_handler]
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
            <title>Study: {}</title>
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
            </div>
            
            <h1>{}</h1>
            <p><strong>Goal:</strong> {}</p>
            
            <div class="cards">
                {}
            </div>
        </body>
        </html>
    "#,
        goal.description,
        goal.description,
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
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize system
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let system = LearningSystem::new();
    
    // Create password hash (change 'your_password_here' to your desired password)
    let password_hash = hash("your_password_here", DEFAULT_COST)?;

    // Create shared state
    let state = Arc::new(Mutex::new(AppState {
        learning_system: system,
        login_attempts: HashMap::new(),
        password_hash,
    }));

    // Build router
    let app = Router::new()
        .route("/", get(show_login))
        .route("/login", post(handle_login))
        .route("/dashboard", get(show_dashboard))
        .route("/goals/new", get(show_goal_form).post(handle_goal_creation))
        .route("/study/:goal_id", get(show_study_page))
        .route("/study/:goal_id/submit/:card_id", post(handle_answer_submission))
        .with_state(state);

    println!("Server running on http://localhost:3000");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app.into_make_service()).await?;

    Ok(())
}
