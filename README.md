# Rusty Paragraphizer

**Rusty Paragraphizer** groups semantically related sentences into coherent paragraphs using embeddings and a C99-style boundary detector.
Useful for vector database preprocessing and transcript segmentation.

---

## Usage Guide

To include this crate in your project, add it to your dependencies:

```bash
cargo add --git https://github.com/sindre0830/rusty-paragraphizer.git --tag v0.1.0 rusty-paragraphizer
```

Or manually in your `Cargo.toml`:

```toml
[dependencies]
rusty-paragraphizer = { git = "https://github.com/sindre0830/rusty-paragraphizer.git", tag = "v0.1.0" }
```

### Example

```rust
use rusty_paragraphizer::ParagraphGrouper;

fn main() -> anyhow::Result<()> {
    let sentences = vec![
        "Humanity’s fascination with space has driven exploration beyond our planet.".to_string(),
        "The Apollo missions of the 20th century marked the first steps onto another world.".to_string(),
        "Today, private companies are rekindling that ambition through commercial spaceflight.".to_string(),
        "Reusable rockets have drastically reduced launch costs, enabling more frequent missions.".to_string(),
        "This new era is often described as the dawn of space industrialization.".to_string(),

        "Satellites now serve as the backbone of global communication systems.".to_string(),
        "They provide navigation, weather forecasting, and even agricultural monitoring.".to_string(),
        "Low-Earth-orbit constellations promise faster internet in remote areas.".to_string(),

        "Climate technology has become one of the fastest-growing sectors for investment.".to_string(),
        "Companies are developing advanced sensors to track carbon emissions.".to_string(),

        "Artificial intelligence and robotics play a major role in these advancements.".to_string(),
        "Autonomous drones can inspect power lines, oil rigs, and wind turbines.".to_string(),
        "Machine learning models optimize energy grids by predicting demand surges.".to_string(),
        "As automation scales, the focus shifts toward ethical deployment and workforce transition.".to_string(),
    ];

    let paragraphs = ParagraphGrouper::new()
        .split_threshold(0.55)
        .min_sentences(2)
        .rank_window(2)
        .block_depth(2)
        .group(sentences)?;

    for (i, p) in paragraphs.iter().enumerate() {
        println!("Paragraph {i}: {}", p);
    }

    Ok(())
}
```

---

## API Overview

TBA

### Core Functions

TBA

---

## Development Guide

### Commands

| Command            | Description                             | Example                                                    |
| ------------------ | --------------------------------------- | ---------------------------------------------------------- |
| **Build**          | Compiles the crate in release mode      | `cargo build --release`                                    |
| **Run Tests**      | Executes all unit tests                 | `cargo test`                                               |
| **Lint (Clippy)**  | Checks for style and performance issues | `cargo clippy --all-targets --all-features -- -D warnings` |
| **Format Code**    | Formats the entire codebase             | `cargo fmt`                                                |
| **Doc Generation** | Builds local documentation              | `cargo doc --open`                                         |

---

### Upgrading Dependencies

To upgrade all dependencies to the latest compatible versions:

```bash
cargo update
```

Or for a specific crate:

```bash
cargo update -p crate-name
```
