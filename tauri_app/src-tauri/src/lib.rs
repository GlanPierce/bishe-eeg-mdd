use std::{
    collections::HashMap,
    env,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use serde::Serialize;

#[derive(Serialize)]
struct AppModelEntry {
    key: &'static str,
    label: String,
    folder: &'static str,
    path: String,
    console_only: bool,
    disabled: bool,
    badge: Option<&'static str>,
}

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

fn model_folder_root(repo_root: &Path) -> PathBuf {
    repo_root.join("outputs").join("app_models")
}

fn model_profiles() -> [(&'static str, &'static str, &'static str, bool, Option<&'static str>); 3] {
    [
        (
            "clean",
            "ExplicitRegionTemporalWeightedStarGNN（61人TASK主基准）",
            "ExplicitRegionTemporalWeightedStarGNN",
            false,
            None,
        ),
        (
            "targeted_repair",
            "ExplicitRegionTemporalWeightedStarGNN（定向伪迹修复变体）",
            "ExplicitRegionTemporalWeightedStarGNN_targeted_artifact_repair",
            false,
            None,
        ),
        (
            "taskonly61_multistate_arch",
            "Multistate explicit region-temporal weighted-star model（TASK+EC+EO）",
            "Multistate_explicit_region_temporal_weighted_star_model",
            true,
            None,
        ),
    ]
}

#[tauri::command]
fn list_model_folder() -> Result<Vec<AppModelEntry>, String> {
    let repo_root = find_repo_root()?;
    let root = model_folder_root(&repo_root);
    fs::create_dir_all(&root).map_err(|err| format!("Failed to create model folder: {err}"))?;
    let models = model_profiles()
        .into_iter()
        .filter_map(|(key, _label, folder, console_only, badge)| {
            let path = root.join(folder).join("model.pkl");
            path.exists().then(|| AppModelEntry {
                key,
                label: folder.to_string(),
                folder,
                path: path.to_string_lossy().to_string(),
                console_only,
                disabled: console_only,
                badge,
            })
        })
        .collect();
    Ok(models)
}

#[tauri::command]
fn run_inference(
    edf_path: String,
    model_profile: Option<String>,
    custom_model_path: Option<String>,
    edf_states: Option<HashMap<String, String>>,
) -> Result<serde_json::Value, String> {
    if edf_path.trim().is_empty() && edf_states.as_ref().map_or(true, HashMap::is_empty) {
        return Err("No EDF path provided.".to_string());
    }

    let repo_root = find_repo_root()?;
    let script = repo_root.join("src").join("app_backend").join("infer_edf.py");
    let profile = model_profile.clone().unwrap_or_else(|| "clean".to_string());
    let allowed = model_profiles()
        .into_iter()
        .any(|(key, _, folder, _, _)| key == profile && model_folder_root(&repo_root).join(folder).join("model.pkl").exists());
    if custom_model_path.as_ref().map_or(true, |path| path.trim().is_empty()) && !allowed {
        return Err(format!(
            "Model artifact is not available in outputs/app_models for profile: {profile}"
        ));
    }
    let mut command = Command::new(python_path(&repo_root));
    command
        .arg(script)
        .arg("--edf")
        .arg(edf_path)
        .arg("--model-profile")
        .arg(profile)
        .arg("--require-artifact");
    if let Some(path) = custom_model_path {
        if !path.trim().is_empty() {
            command.arg("--custom-model-path").arg(path);
        }
    }
    if let Some(states) = edf_states {
        for state in ["TASK", "EC", "EO"] {
            if let Some(path) = states.get(state) {
                if !path.trim().is_empty() {
                    command.arg("--edf-state").arg(format!("{state}={path}"));
                }
            }
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

#[tauri::command]
fn open_model_folder() -> Result<String, String> {
    let repo_root = find_repo_root()?;
    let folder = model_folder_root(&repo_root);
    fs::create_dir_all(&folder).map_err(|err| format!("Failed to create model folder: {err}"))?;
    Command::new("explorer")
        .arg(&folder)
        .spawn()
        .map_err(|err| format!("Failed to open model folder: {err}"))?;
    Ok(folder.to_string_lossy().to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![run_inference, open_model_folder, list_model_folder])
        .run(tauri::generate_context!())
        .expect("error while running Tauri application");
}
