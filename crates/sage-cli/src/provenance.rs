use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileFingerprint {
    pub path: PathBuf,
    pub size: u64,
    pub modified_unix_seconds: Option<u64>,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BaselineManifest {
    pub schema_version: u32,
    pub created_unix_seconds: u64,
    pub source_commit: Option<String>,
    pub source_dirty: Option<bool>,
    pub status: String,
    pub files: Vec<FileFingerprint>,
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path)
        .with_context(|| format!("failed to open {} for hashing", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let n = reader
            .read(&mut buffer)
            .with_context(|| format!("failed while hashing {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn fingerprint(path: &Path) -> Result<FileFingerprint> {
    let metadata = path
        .metadata()
        .with_context(|| format!("failed to stat {}", path.display()))?;
    anyhow::ensure!(metadata.is_file(), "{} is not a file", path.display());
    let modified_unix_seconds = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs());
    Ok(FileFingerprint {
        path: path.to_path_buf(),
        size: metadata.len(),
        modified_unix_seconds,
        sha256: sha256_file(path)?,
    })
}

pub fn collect_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    fn visit(path: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
        if path.is_file() {
            output.push(path.to_path_buf());
            return Ok(());
        }
        anyhow::ensure!(
            path.is_dir(),
            "baseline path does not exist: {}",
            path.display()
        );
        let mut children = std::fs::read_dir(path)
            .with_context(|| format!("failed to read directory {}", path.display()))?
            .collect::<std::io::Result<Vec<_>>>()?;
        children.sort_by_key(|entry| entry.file_name());
        for child in children {
            visit(&child.path(), output)?;
        }
        Ok(())
    }

    let mut files = Vec::new();
    for path in paths {
        visit(path, &mut files)?;
    }
    files.sort();
    files.dedup();
    Ok(files)
}

fn git_value(repo: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?.trim().to_owned();
    (!value.is_empty()).then_some(value)
}

pub fn source_state(repo: &Path) -> (Option<String>, Option<bool>) {
    let commit = git_value(repo, &["rev-parse", "HEAD"]);
    let dirty = git_value(repo, &["status", "--porcelain"]).map(|status| !status.is_empty());
    (commit, dirty)
}

pub fn freeze_baseline(
    paths: &[PathBuf],
    source_repo: &Path,
    status: impl Into<String>,
) -> Result<BaselineManifest> {
    let files = collect_files(paths)?;
    let fingerprints = files
        .iter()
        .map(|path| fingerprint(path))
        .collect::<Result<Vec<_>>>()?;
    let (source_commit, source_dirty) = source_state(source_repo);
    Ok(BaselineManifest {
        schema_version: 1,
        created_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        source_commit,
        source_dirty,
        status: status.into(),
        files: fingerprints,
    })
}

pub fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .with_context(|| format!("failed to create {}", parent.display()))?;
    let temporary = path.with_extension(format!(
        "{}.tmp",
        path.extension().and_then(|x| x.to_str()).unwrap_or("json")
    ));
    let bytes = serde_json::to_vec_pretty(value)?;
    std::fs::write(&temporary, bytes)
        .with_context(|| format!("failed to write {}", temporary.display()))?;
    std::fs::rename(&temporary, path).with_context(|| {
        format!(
            "failed to atomically replace {} with {}",
            path.display(),
            temporary.display()
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_is_stable() {
        let path = std::env::temp_dir().join(format!(
            "sage-provenance-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, b"sage\n").unwrap();
        assert_eq!(
            sha256_file(&path).unwrap(),
            "8626c5cb340788b38c2baf2a2f5396b0d29a5ce9925ea66378b4c17a6e98cbb2"
        );
        std::fs::remove_file(path).unwrap();
    }
}
