//! Portable, mutation-checked identities for file- and directory-backed inputs.

use crate::provenance::sha256_file;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{File, Metadata};
use std::io::{BufReader, Read};
use std::path::{Component, Path, PathBuf};
use std::time::SystemTime;

pub const DIRECTORY_IDENTITY_SCHEMA: &str = "sage-input-directory-content-v1";
const DIRECTORY_IDENTITY_DOMAIN: &[u8] = b"sage-input-path-identity\0directory-content\0v1";

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InputPathKind {
    #[default]
    RegularFile,
    Directory,
    RemoteSource,
}

impl InputPathKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RegularFile => "regular_file",
            Self::Directory => "directory",
            Self::RemoteSource => "remote_source",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct InputPathIdentity {
    pub kind: InputPathKind,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub directory_schema: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regular_file_count: Option<u64>,
    pub total_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StabilityMetadata {
    len: u64,
    modified: Option<SystemTime>,
    #[cfg(unix)]
    dev: u64,
    #[cfg(unix)]
    ino: u64,
    #[cfg(unix)]
    mtime: i64,
    #[cfg(unix)]
    mtime_nsec: i64,
    #[cfg(unix)]
    ctime: i64,
    #[cfg(unix)]
    ctime_nsec: i64,
}

impl StabilityMetadata {
    fn from_file(metadata: &Metadata) -> Result<Self> {
        anyhow::ensure!(
            metadata.is_file(),
            "input entry is no longer a regular file"
        );
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        Ok(Self {
            len: metadata.len(),
            modified: metadata.modified().ok(),
            #[cfg(unix)]
            dev: metadata.dev(),
            #[cfg(unix)]
            ino: metadata.ino(),
            #[cfg(unix)]
            mtime: metadata.mtime(),
            #[cfg(unix)]
            mtime_nsec: metadata.mtime_nsec(),
            #[cfg(unix)]
            ctime: metadata.ctime(),
            #[cfg(unix)]
            ctime_nsec: metadata.ctime_nsec(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DirectoryEntrySnapshot {
    relative_path: PathBuf,
    normalized_relative_path: String,
    metadata: StabilityMetadata,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HashEvent {
    InventoryFrozen,
    FirstContentChunk,
}

fn normalized_relative_path(path: &Path) -> Result<String> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(value) => {
                let value = value.to_str().with_context(|| {
                    format!("input path is not valid UTF-8: {}", path.display())
                })?;
                anyhow::ensure!(
                    !value.is_empty() && value != "." && value != "..",
                    "input path contains a non-portable component: {}",
                    path.display()
                );
                parts.push(value);
            }
            _ => anyhow::bail!(
                "input path is not normalized root-relative: {}",
                path.display()
            ),
        }
    }
    anyhow::ensure!(
        !parts.is_empty(),
        "directory entry has an empty relative path"
    );
    Ok(parts.join("/"))
}

fn scan_directory(root: &Path) -> Result<Vec<DirectoryEntrySnapshot>> {
    let root_metadata = std::fs::symlink_metadata(root)
        .with_context(|| format!("failed to stat input directory {}", root.display()))?;
    anyhow::ensure!(
        !root_metadata.file_type().is_symlink(),
        "directory-backed input root may not be a symlink: {}",
        root.display()
    );
    anyhow::ensure!(
        root_metadata.is_dir(),
        "directory-backed input is not a directory: {}",
        root.display()
    );

    let mut pending = vec![PathBuf::new()];
    let mut files = Vec::new();
    while let Some(relative_directory) = pending.pop() {
        let directory = root.join(&relative_directory);
        let entries = std::fs::read_dir(&directory)
            .with_context(|| format!("failed to read input directory {}", directory.display()))?
            .collect::<std::io::Result<Vec<_>>>()
            .with_context(|| {
                format!(
                    "failed to enumerate input directory {}",
                    directory.display()
                )
            })?;
        for entry in entries {
            let relative = relative_directory.join(entry.file_name());
            let path = root.join(&relative);
            let metadata = std::fs::symlink_metadata(&path)
                .with_context(|| format!("failed to stat input entry {}", path.display()))?;
            let file_type = metadata.file_type();
            if file_type.is_symlink() {
                anyhow::bail!(
                    "symlink in directory-backed input is prohibited: {}",
                    path.display()
                );
            } else if file_type.is_dir() {
                pending.push(relative);
            } else if file_type.is_file() {
                files.push(DirectoryEntrySnapshot {
                    normalized_relative_path: normalized_relative_path(&relative)?,
                    relative_path: relative,
                    metadata: StabilityMetadata::from_file(&metadata)?,
                });
            } else {
                anyhow::bail!(
                    "unsupported special entry in directory-backed input: {}",
                    path.display()
                );
            }
        }
    }
    files.sort_by(|left, right| {
        left.normalized_relative_path
            .cmp(&right.normalized_relative_path)
    });
    let mut unique = HashSet::with_capacity(files.len());
    for file in &files {
        anyhow::ensure!(
            unique.insert(file.normalized_relative_path.clone()),
            "duplicate normalized relative path in directory-backed input: {}",
            file.normalized_relative_path
        );
    }
    Ok(files)
}

fn frame(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn hash_open_file_stable<F>(
    path: &Path,
    expected: &StabilityMetadata,
    observer: &mut F,
) -> Result<[u8; 32]>
where
    F: FnMut(HashEvent, &Path) -> Result<()>,
{
    let path_before = std::fs::symlink_metadata(path).with_context(|| {
        format!(
            "failed to stat input file before hashing {}",
            path.display()
        )
    })?;
    anyhow::ensure!(
        !path_before.file_type().is_symlink(),
        "input file became a symlink before hashing: {}",
        path.display()
    );
    anyhow::ensure!(
        StabilityMetadata::from_file(&path_before)? == *expected,
        "input file changed after directory inventory was frozen: {}",
        path.display()
    );
    let file = File::open(path)
        .with_context(|| format!("failed to open input file for hashing {}", path.display()))?;
    anyhow::ensure!(
        StabilityMetadata::from_file(&file.metadata()?)? == *expected,
        "input file changed while it was opened for hashing: {}",
        path.display()
    );
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    let mut observed_first_chunk = false;
    loop {
        let count = reader
            .read(&mut buffer)
            .with_context(|| format!("failed while hashing input file {}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        if !observed_first_chunk {
            observed_first_chunk = true;
            observer(HashEvent::FirstContentChunk, path)?;
        }
    }
    let file = reader.into_inner();
    anyhow::ensure!(
        StabilityMetadata::from_file(&file.metadata()?)? == *expected,
        "input file changed during content hashing: {}",
        path.display()
    );
    let path_after = std::fs::symlink_metadata(path)
        .with_context(|| format!("failed to stat input file after hashing {}", path.display()))?;
    anyhow::ensure!(
        !path_after.file_type().is_symlink()
            && StabilityMetadata::from_file(&path_after)? == *expected,
        "input path changed during content hashing: {}",
        path.display()
    );
    Ok(hasher.finalize().into())
}

fn directory_identity_with_observer<F>(path: &Path, mut observer: F) -> Result<InputPathIdentity>
where
    F: FnMut(HashEvent, &Path) -> Result<()>,
{
    let inventory = scan_directory(path)?;
    observer(HashEvent::InventoryFrozen, path)?;
    let regular_file_count =
        u64::try_from(inventory.len()).context("too many directory entries")?;
    anyhow::ensure!(
        regular_file_count > 0,
        "directory-backed input contains no regular files: {}",
        path.display()
    );
    let total_bytes = inventory.iter().try_fold(0_u64, |sum, entry| {
        sum.checked_add(entry.metadata.len)
            .context("directory-backed input byte count overflow")
    })?;

    let mut hasher = Sha256::new();
    frame(&mut hasher, DIRECTORY_IDENTITY_DOMAIN);
    frame(&mut hasher, DIRECTORY_IDENTITY_SCHEMA.as_bytes());
    frame(&mut hasher, &regular_file_count.to_le_bytes());
    frame(&mut hasher, &total_bytes.to_le_bytes());
    for entry in &inventory {
        let content = hash_open_file_stable(
            &path.join(&entry.relative_path),
            &entry.metadata,
            &mut observer,
        )?;
        frame(&mut hasher, b"regular_file");
        frame(&mut hasher, entry.normalized_relative_path.as_bytes());
        frame(&mut hasher, &entry.metadata.len.to_le_bytes());
        frame(&mut hasher, &content);
    }
    let final_inventory = scan_directory(path)?;
    anyhow::ensure!(
        final_inventory == inventory,
        "directory-backed input inventory changed during hashing: {}",
        path.display()
    );

    Ok(InputPathIdentity {
        kind: InputPathKind::Directory,
        sha256: format!("{:x}", hasher.finalize()),
        directory_schema: Some(DIRECTORY_IDENTITY_SCHEMA.into()),
        regular_file_count: Some(regular_file_count),
        total_bytes,
    })
}

pub fn input_path_identity(path: &Path) -> Result<InputPathIdentity> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("failed to stat input path {}", path.display()))?;
    if metadata.file_type().is_symlink() {
        let resolved = path
            .canonicalize()
            .with_context(|| format!("failed to resolve input symlink {}", path.display()))?;
        let resolved_metadata = resolved
            .metadata()
            .with_context(|| format!("failed to stat resolved input {}", resolved.display()))?;
        anyhow::ensure!(
            resolved_metadata.is_file(),
            "symlinked directory-backed inputs are prohibited: {}",
            path.display()
        );
        return Ok(InputPathIdentity {
            kind: InputPathKind::RegularFile,
            sha256: sha256_file(&resolved)?,
            directory_schema: None,
            regular_file_count: None,
            total_bytes: resolved_metadata.len(),
        });
    }
    if metadata.is_file() {
        return Ok(InputPathIdentity {
            kind: InputPathKind::RegularFile,
            sha256: sha256_file(path)?,
            directory_schema: None,
            regular_file_count: None,
            total_bytes: metadata.len(),
        });
    }
    if metadata.is_dir() {
        return directory_identity_with_observer(path, |_, _| Ok(()));
    }
    anyhow::bail!("unsupported input path type: {}", path.display())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "sage-input-path-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn fixture(root: &Path, reverse: bool) {
        fs::create_dir_all(root.join("nested/deeper")).unwrap();
        let entries = [
            ("alpha.bin", b"alpha".as_slice()),
            ("nested/beta.bin", b"beta".as_slice()),
            ("nested/deeper/gamma.bin", b"gamma".as_slice()),
        ];
        let iter: Box<dyn Iterator<Item = &(&str, &[u8])>> = if reverse {
            Box::new(entries.iter().rev())
        } else {
            Box::new(entries.iter())
        };
        for (relative, bytes) in iter {
            fs::write(root.join(relative), bytes).unwrap();
        }
    }

    #[test]
    fn regular_file_hash_preserves_legacy_sha256() {
        let root = test_root("regular");
        fs::create_dir_all(&root).unwrap();
        let file = root.join("input.mzML");
        fs::write(&file, b"sage\n").unwrap();
        let identity = input_path_identity(&file).unwrap();
        assert_eq!(identity.kind, InputPathKind::RegularFile);
        assert_eq!(
            identity.sha256,
            "8626c5cb340788b38c2baf2a2f5396b0d29a5ce9925ea66378b4c17a6e98cbb2"
        );
        assert_eq!(identity.sha256, sha256_file(&file).unwrap());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn directory_identity_is_relocation_and_order_invariant() {
        let first = test_root("first");
        let second = test_root("second");
        fixture(&first, false);
        fixture(&second, true);
        let left = input_path_identity(&first).unwrap();
        let right = input_path_identity(&second).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.regular_file_count, Some(3));
        assert_eq!(left.total_bytes, 14);
        fs::remove_dir_all(first).unwrap();
        fs::remove_dir_all(second).unwrap();
    }

    #[test]
    fn directory_content_add_remove_and_rename_change_identity() {
        let root = test_root("changes");
        fixture(&root, false);
        let baseline = input_path_identity(&root).unwrap();
        fs::write(root.join("alpha.bin"), b"ALPHA").unwrap();
        assert_ne!(baseline, input_path_identity(&root).unwrap());
        fs::write(root.join("alpha.bin"), b"alpha").unwrap();
        fs::write(root.join("added.bin"), b"added").unwrap();
        let added = input_path_identity(&root).unwrap();
        assert_ne!(baseline, added);
        fs::remove_file(root.join("added.bin")).unwrap();
        assert_eq!(baseline, input_path_identity(&root).unwrap());
        fs::rename(root.join("alpha.bin"), root.join("renamed.bin")).unwrap();
        assert_ne!(baseline, input_path_identity(&root).unwrap());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn empty_directory_fails_closed() {
        let root = test_root("empty");
        fs::create_dir_all(&root).unwrap();
        assert!(input_path_identity(&root)
            .unwrap_err()
            .to_string()
            .contains("contains no regular files"));
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn directory_symlinks_and_special_files_fail_closed() {
        use std::os::unix::fs::symlink;
        let root = test_root("unsupported");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("file"), b"content").unwrap();
        symlink(root.join("file"), root.join("link")).unwrap();
        assert!(input_path_identity(&root)
            .unwrap_err()
            .to_string()
            .contains("symlink"));
        fs::remove_file(root.join("link")).unwrap();
        let fifo = root.join("fifo");
        assert!(std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success());
        assert!(input_path_identity(&root)
            .unwrap_err()
            .to_string()
            .contains("unsupported special entry"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn directory_inventory_drift_fails_closed() {
        let root = test_root("inventory-drift");
        fixture(&root, false);
        let added = root.join("appeared.bin");
        let error = directory_identity_with_observer(&root, |event, _| {
            if event == HashEvent::InventoryFrozen && !added.exists() {
                fs::write(&added, b"late")?;
            }
            Ok(())
        })
        .unwrap_err();
        assert!(error.to_string().contains("inventory changed"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn directory_content_mutation_while_hashing_fails_closed() {
        let root = test_root("content-drift");
        fs::create_dir_all(&root).unwrap();
        let file = root.join("large.bin");
        fs::write(&file, vec![b'x'; 2 * 1024 * 1024]).unwrap();
        let mut mutated = false;
        let error = directory_identity_with_observer(&root, |event, path| {
            if event == HashEvent::FirstContentChunk && !mutated {
                let mut bytes = fs::read(path)?;
                bytes[0] = b'y';
                fs::write(path, bytes)?;
                mutated = true;
            }
            Ok(())
        })
        .unwrap_err();
        assert!(error.to_string().contains("changed during content hashing"));
        fs::remove_dir_all(root).unwrap();
    }
}
