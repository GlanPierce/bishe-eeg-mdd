use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

fn find_repo_root() -> Result<PathBuf, String> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.parent().and_then(Path::parent).map(Path::to_path_buf),
        env::current_dir().ok(),
    ];

    for candidate in candidates.into_iter().flatten() {
        for ancestor in candidate.ancestors() {
            if ancestor.join("src").join("app_backend").join("infer_edf.py").exists() {
                return Ok(ancestor.to_path_buf());
            }
        }
    }

    Err("Cannot locate repository root containing src/app_backend/infer_edf.py.".to_string())
}

fn python_path(repo_root: &Path) -> PathBuf {
    let venv_python = repo_root.join(".venv").join("Scripts").join("python.exe");
    if venv_python.exists() {
        venv_python
    } else {
        PathBuf::from("python")
    }
}

#[tauri::command]
fn run_inference(
    edf_path: String,
    model_profile: Option<String>,
    custom_model_path: Option<String>,
) -> Result<serde_json::Value, String> {
    if edf_path.trim().is_empty() {
        return Err("No EDF path provided.".to_string());
    }

    let repo_root = find_repo_root()?;
    let script = repo_root.join("src").join("app_backend").join("infer_edf.py");
    let mut command = Command::new(python_path(&repo_root));
    command
        .arg(script)
        .arg("--edf")
        .arg(edf_path)
        .arg("--model-profile")
        .arg(model_profile.unwrap_or_else(|| "clean".to_string()))
        .arg("--trust-cache");
    if let Some(path) = custom_model_path {
        if !path.trim().is_empty() {
            command.arg("--custom-model-path").arg(path);
        }
    }
    let output = command
        .current_dir(&repo_root)
        .env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .output()
        .map_err(|err| format!("Failed to start Python inference: {err}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let message = stderr
            .lines()
            .last()
            .and_then(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .and_then(|value| value.get("error").and_then(|error| error.as_str()).map(str::to_string))
            .unwrap_or_else(|| {
                let trimmed = stderr.trim();
                if trimmed.is_empty() {
                    "Inference failed.".to_string()
                } else {
                    trimmed.to_string()
                }
            });
        return Err(message);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let last_line = stdout
        .lines()
        .last()
        .ok_or_else(|| "Inference returned no JSON output.".to_string())?;
    serde_json::from_str(last_line).map_err(|err| format!("Invalid inference JSON: {err}"))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![run_inference])
        .run(tauri::generate_context!())
        .expect("error while running Tauri application");
}
