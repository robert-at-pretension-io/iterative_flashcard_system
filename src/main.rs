use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use uuid::Uuid;

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let mut system = LearningSystem::new();

    // Example workflow
    let goal = system.discover_goal(&api_key, "Learn Rust ownership system").await?;
    println!("Goal established: {:?}", goal);

    let cards = system.generate_cards_for_goal(&api_key, goal.id).await?;
    println!("Generated {} cards", cards.len());

    // Simulate user interaction
    for card in &cards {
        println!("\nQuestion: {}", card.question);
        // In a real application, you would get user input here
        let user_response = "The ownership system ensures memory safety at compile time";
        let discussion = system.evaluate_response(&api_key, card.id, user_response).await?;
        println!("Feedback: {}", discussion.critique);
    }

    system.save("learning_progress.json")?;
    println!("Progress saved!");

    Ok(())
}
