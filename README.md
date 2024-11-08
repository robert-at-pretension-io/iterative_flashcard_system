# Iterative Flashcard System

An intelligent flashcard system that uses AI to help create and manage personalized learning goals and flashcards.

## Features

- 🎯 AI-powered learning goal discovery and refinement
- 🔄 Dynamic flashcard generation based on your learning goals
- ⏱️ Spaced repetition system for optimal learning
- 🤖 Interactive study sessions with AI-powered feedback
- 📊 Knowledge graph visualization of your learning progress
- 📚 Curriculum planning and tracking
- 📈 Performance analytics and weak topic identification

## Prerequisites

### Installing Rust

1. Install Rust using rustup (recommended method):

**Unix-like OS (Linux, macOS):**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Windows:**
1. Download and run [rustup-init.exe](https://rustup.rs/)
2. Follow the on-screen instructions
3. You may need to install Visual Studio C++ Build Tools

After installation:
1. Restart your terminal
2. Verify installation:
```bash
rustc --version
cargo --version
```

If you need to update Rust later:
```bash
rustup update
```

### Other Prerequisites
- Rust (1.70 or later)
- OpenAI API key (gpt-4o-mini access recommended)
- 512MB RAM minimum
- 1GB free disk space
- Internet connection

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/robert-at-pretension-io/iterative_flashcard_system
cd iterative-flashcard-system
```

2. Set up your OpenAI API key:

Linux/macOS:
```bash
echo "export OPENAI_API_KEY='your-api-key-here'" >> ~/.bashrc
source ~/.bashrc
```

Windows (PowerShell):
```powershell
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "your-api-key-here", "User")
```

3. Build and run:
```bash
cargo build --release
cargo run --release
```

4. Open your browser:
```
http://localhost:3000
```

Default password: `your_password_here`

## Project Structure

```
.
├── src/
│   └── main.rs          # Main application code
├── tests/               # Integration tests
├── docs/               # Documentation
├── learning_system.json # Data storage
└── Cargo.toml          # Dependencies
```

## Configuration

### Password Management

For development:
```rust
let password_hash = hash("your_new_password_here", DEFAULT_COST);
```


## Troubleshooting

### Common Issues

1. "Failed to connect to OpenAI API"
   - Check API key is set correctly
   - Verify internet connection
   - Ensure API key has sufficient credits

2. "Database errors"
   - Check disk permissions
   - Verify JSON file integrity
   - Clear corrupted data: `rm learning_system.json`

3. "Server won't start"
   - Check port availability
   - Verify Rust version
   - Clear target directory: `cargo clean`


### Code Style

- Follow Rust style guidelines
- Use meaningful variable names
- Add comments for complex logic
- Include unit tests for new features

## License

MIT License - see [LICENSE](LICENSE) for details

## Support

- good luck

## Acknowledgments

- OpenAI for GPT API
- Rust community
- http://aider.chat
