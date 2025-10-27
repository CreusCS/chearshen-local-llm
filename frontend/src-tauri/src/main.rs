// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod grpc_client;

use std::path::Path;

use async_stream::stream;
use grpc_client::proto;
use grpc_client::GrpcState;
use serde::Serialize;
use serde_json::Value;
use tauri::State;
use tokio::fs;
use tokio::io::{AsyncReadExt, BufReader};

const MAX_VIDEO_BYTES: u64 = 100 * 1024 * 1024;
const CHUNK_SIZE: usize = 512 * 1024;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TranscriptionPayload {
  success: bool,
  session_id: String,
  transcription: String,
  error_message: Option<String>,
  video_filename: String,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
struct ActionPlanPayload {
  action_type: String,
  status: String,
  parameters: Value,
  missing_params: Vec<String>,
  clarification_question: Option<String>,
  confirmation_message: Option<String>,
  requires_user_input: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ChatPayload {
  success: bool,
  response: String,
  error_message: Option<String>,
  session_id: String,
  action_plan: Option<ActionPlanPayload>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ChatHistoryMessagePayload {
  id: String,
  role: String,
  content: String,
  timestamp: String,
  session_id: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ChatHistoryPayload {
  success: bool,
  messages: Vec<ChatHistoryMessagePayload>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeneratePdfPayload {
  success: bool,
  pdf_data: Vec<u8>,
  filename: String,
  error_message: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SessionContextPayload {
  success: bool,
  error_message: Option<String>,
  session: Option<SessionMetadataPayload>,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
struct SessionMetadataPayload {
  session_id: String,
  video_filename: String,
  transcription: String,
  summary: String,
  created_at: String,
  updated_at: String,
}

#[tauri::command]
async fn transcribe_video(
  state: State<'_, GrpcState>,
  session_id: String,
  file_path: String,
) -> Result<TranscriptionPayload, String> {
  let metadata = fs::metadata(&file_path)
    .await
    .map_err(|err| format!("Unable to read file metadata: {err}"))?;

  if metadata.len() > MAX_VIDEO_BYTES {
    return Err("File size must be less than 100MB".into());
  }

  let filename = Path::new(&file_path)
    .file_name()
    .and_then(|name| name.to_str())
    .unwrap_or("video.mp4")
    .to_string();

  let client = state
    .get_client()
    .await
    .map_err(|err| format!("Failed to connect to backend: {err}"))?;

  let file = fs::File::open(&file_path)
    .await
    .map_err(|err| format!("Unable to open file: {err}"))?;
  let mut reader = BufReader::new(file);

  let session_id_for_chunks = session_id.clone();
  let filename_for_chunks = filename.clone();

  let outbound = stream! {
    loop {
      let mut buffer = vec![0u8; CHUNK_SIZE];
      match reader.read(&mut buffer).await {
        Ok(0) => break,
        Ok(bytes_read) => {
          buffer.truncate(bytes_read);
          yield Ok(proto::VideoUploadChunk {
            data: buffer,
            filename: filename_for_chunks.clone(),
            session_id: session_id_for_chunks.clone(),
          });
        }
        Err(err) => {
          yield Err(tonic::Status::internal(err.to_string()));
          break;
        }
      }
    }
  };

  let mut locked_client = client.lock().await;

  let response = locked_client
    .transcribe_video(outbound)
    .await
    .map_err(|err| format!("Transcription request failed: {err}"))?
    .into_inner();

  Ok(TranscriptionPayload {
    success: response.success,
    session_id: response.session_id,
    transcription: response.transcription,
    error_message: if response.error_message.is_empty() {
      None
    } else {
      Some(response.error_message)
    },
    video_filename: response.video_filename,
  })
}

#[tauri::command]
async fn send_chat_message(
  state: State<'_, GrpcState>,
  session_id: String,
  message: String,
  context: Option<String>,
) -> Result<ChatPayload, String> {
  let client = state
    .get_client()
    .await
    .map_err(|err| format!("Failed to connect to backend: {err}"))?;

  let request = proto::ChatRequest {
    session_id: session_id.clone(),
    message,
    context: context.unwrap_or_default(),
  };

  let mut locked_client = client.lock().await;

  let response = locked_client
    .chat(request)
    .await
    .map_err(|err| format!("Chat request failed: {err}"))?
    .into_inner();

  let action_plan = response.action_plan.map(|plan| {
    let parameters = if plan.parameters_json.is_empty() {
      Value::Object(Default::default())
    } else {
      serde_json::from_str(&plan.parameters_json).unwrap_or_else(|_| Value::Object(Default::default()))
    };
    ActionPlanPayload {
      action_type: plan.action_type,
      status: plan.status,
      parameters,
      missing_params: plan.missing_params,
      clarification_question: if plan.clarification_question.is_empty() {
        None
      } else {
        Some(plan.clarification_question)
      },
      confirmation_message: if plan.confirmation_message.is_empty() {
        None
      } else {
        Some(plan.confirmation_message)
      },
      requires_user_input: plan.requires_user_input,
    }
  });

  Ok(ChatPayload {
    success: response.success,
    response: response.response_text,
    error_message: if response.error_message.is_empty() {
      None
    } else {
      Some(response.error_message)
    },
    session_id: response.session_id,
    action_plan,
  })
}

#[tauri::command]
async fn get_chat_history(
  state: State<'_, GrpcState>,
  session_id: String,
  limit: Option<u32>,
) -> Result<ChatHistoryPayload, String> {
  let client = state
    .get_client()
    .await
    .map_err(|err| format!("Failed to connect to backend: {err}"))?;

  let mut locked_client = client.lock().await;

  let response = locked_client
    .get_chat_history(proto::ChatHistoryRequest {
      session_id: session_id.clone(),
      limit: limit.unwrap_or(50) as i32,
    })
    .await
    .map_err(|err| format!("Chat history request failed: {err}"))?
    .into_inner();

  let messages = response
    .messages
    .into_iter()
    .map(|msg| ChatHistoryMessagePayload {
      id: msg.id,
      role: msg.role,
      content: msg.content,
      timestamp: msg.timestamp,
      session_id: session_id.clone(),
    })
    .collect();

  Ok(ChatHistoryPayload {
    success: true,
    messages,
  })
}

#[tauri::command]
async fn clear_chat_history(state: State<'_, GrpcState>, session_id: String) -> Result<(), String> {
  let client = state
    .get_client()
    .await
    .map_err(|err| format!("Failed to connect to backend: {err}"))?;

  let mut locked_client = client.lock().await;

  locked_client
    .clear_chat_history(proto::SessionRequest { session_id })
    .await
    .map_err(|err| format!("Clear history failed: {err}"))?
    .into_inner();

  Ok(())
}

#[tauri::command]
async fn generate_pdf(
  state: State<'_, GrpcState>,
  session_id: String,
  content: String,
  title: String,
) -> Result<GeneratePdfPayload, String> {
  let client = state
    .get_client()
    .await
    .map_err(|err| format!("Failed to connect to backend: {err}"))?;

  let mut locked_client = client.lock().await;

  let response = locked_client
    .generate_pdf(proto::GeneratePdfRequest {
      session_id,
      content,
      title,
    })
    .await
    .map_err(|err| format!("PDF generation failed: {err}"))?
    .into_inner();

  Ok(GeneratePdfPayload {
    success: response.success,
    pdf_data: response.pdf_data,
    filename: response.filename,
    error_message: if response.error_message.is_empty() {
      None
    } else {
      Some(response.error_message)
    },
  })
}

#[tauri::command]
async fn get_session_context(
  state: State<'_, GrpcState>,
  session_id: String,
) -> Result<SessionContextPayload, String> {
  let client = state
    .get_client()
    .await
    .map_err(|err| format!("Failed to connect to backend: {err}"))?;

  let mut locked_client = client.lock().await;

  let response = locked_client
    .get_session_context(proto::SessionRequest { session_id })
    .await
    .map_err(|err| format!("Session context request failed: {err}"))?
    .into_inner();

  let session = if response.success {
    Some(SessionMetadataPayload {
      session_id: response.session_id,
      video_filename: response.video_filename,
      transcription: response.transcription,
      summary: response.summary,
      created_at: response.created_at,
      updated_at: response.updated_at,
    })
  } else {
    None
  };

  Ok(SessionContextPayload {
    success: response.success,
    error_message: if response.error_message.is_empty() {
      None
    } else {
      Some(response.error_message)
    },
    session,
  })
}

fn main() {
  let endpoint = std::env::var("VIDEO_ANALYZER_ENDPOINT")
    .unwrap_or_else(|_| "http://127.0.0.1:50051".to_string());

  tauri::Builder::default()
    .manage(GrpcState::new(endpoint))
    .invoke_handler(tauri::generate_handler![
      transcribe_video,
      send_chat_message,
      get_chat_history,
      clear_chat_history,
      generate_pdf,
      get_session_context
    ])
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
