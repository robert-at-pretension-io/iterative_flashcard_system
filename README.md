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

- Rust (1.70 or later)
- OpenAI API key (GPT-4 access recommended)
- 512MB RAM minimum
- 1GB free disk space
- Internet connection (1Mbps minimum)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/iterative-flashcard-system.git
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

Default login: `your_password_here`

## Development Setup

1. Install development dependencies:
```bash
rustup component add clippy rustfmt
cargo install cargo-audit cargo-watch
```

2. Run tests:
```bash
cargo test
cargo clippy
```

3. Auto-format code:
```bash
cargo fmt
```

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

For production, use environment variables:
```bash
export IFS_PASSWORD='your-secure-password'
```

### Server Configuration

Default port: 3000
To change:
```bash
export IFS_PORT=8080
```

## Security Considerations

For production deployment:

1. Environment Variables
   - Store sensitive data (API keys, passwords) in env vars
   - Use `.env` files for development only

2. HTTPS Setup
   - Generate SSL certificate
   - Configure reverse proxy (nginx recommended)
   - Enable HTTP/2

3. Session Management
   - Implement JWT tokens
   - Set secure cookie attributes
   - Configure CSRF protection

4. Rate Limiting
   - Login attempts: 3 per 30 minutes
   - API requests: 100 per minute
   - Study sessions: Unlimited

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

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Style

- Follow Rust style guidelines
- Use meaningful variable names
- Add comments for complex logic
- Include unit tests for new features

## License

MIT License - see [LICENSE](LICENSE) for details

## Support

- GitHub Issues: Bug reports and feature requests
- Discussions: General questions and community support
- Email: support@example.com

## Acknowledgments

- OpenAI for GPT API
- Rust community
- All contributors
