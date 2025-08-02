# Contributing to Mental Health Detector

Thank you for your interest in contributing to the Mental Health Detector project! This document provides guidelines for contributing to this ethical AI project focused on mental health support.

## 🛡️ Ethical Guidelines

Before contributing, please understand that this project deals with sensitive mental health data and AI. All contributions must:

1. **Prioritize User Safety**: Any changes must consider potential impact on user well-being
2. **Maintain Privacy**: Respect user confidentiality and data protection principles
3. **Follow Ethical AI**: Ensure fairness, transparency, and accountability
4. **Avoid Harm**: Never introduce features that could cause psychological harm
5. **Professional Standards**: Maintain the highest standards of responsible AI development

## 🤝 Code of Conduct

### Our Pledge
We are committed to making participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards
Examples of behavior that contributes to creating a positive environment include:
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

### Unacceptable Behavior
- Harassment, trolling, or discriminatory comments
- Publishing others' private information without permission
- Misusing mental health terminology or making light of mental health issues
- Any conduct that could harm vulnerable users
- Promoting unethical use of AI or mental health data

## 🚀 Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/midlaj-muhammed/Mental-Health-Detector-AI-Application-Built.git
   cd Mental-Health-Detector-AI-Application-Built
   ```

2. **Set Up Environment**
   ```bash
   # Using Make (recommended)
   make dev-setup
   
   # Or manually
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

4. **Run Tests**
   ```bash
   make test
   # Or: python -m pytest tests/
   ```

### Development Workflow

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or: git checkout -b fix/issue-number
   ```

2. **Make Changes**
   - Follow coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   make test-all  # Run all tests
   make lint      # Check code style
   make security  # Security checks
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new emotion detection feature"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   # Then create a Pull Request on GitHub
   ```

## 📝 Coding Standards

### Python Style Guide
- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use [flake8](https://flake8.pycqa.org/) for linting

### Code Quality Requirements
- **Test Coverage**: Minimum 80% coverage for new code
- **Documentation**: All public functions must have docstrings
- **Type Hints**: Use type hints for function parameters and returns
- **Error Handling**: Proper exception handling with meaningful messages

### Commit Message Format
Use [Conventional Commits](https://www.conventionalcommits.org/) format:
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## 🧪 Testing Guidelines

### Test Types
1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete user workflows
4. **Ethical Tests**: Test for bias, fairness, and safety

### Writing Tests
```python
import pytest
from src.models.emotion_detector import EmotionDetector

def test_emotion_detection_positive_text():
    """Test emotion detection with positive text."""
    detector = EmotionDetector()
    detector.load_model()
    
    result = detector.detect_emotion("I feel amazing today!")
    
    assert result.emotion in ['joy', 'happiness']
    assert result.confidence > 0.7
    assert 'positive' in result.risk_factors
```

### Running Tests
```bash
# Run all tests
make test

# Run specific test file
python -m pytest tests/test_emotion_detector.py

# Run with coverage
make test-coverage

# Run ethical/bias tests
make test-ethics
```

## 📚 Documentation

### Documentation Requirements
- **README Updates**: Update README.md for new features
- **API Documentation**: Document all public APIs
- **Code Comments**: Explain complex logic and algorithms
- **Ethical Considerations**: Document potential ethical implications

### Documentation Style
- Use clear, concise language
- Include code examples
- Explain the "why" not just the "what"
- Consider non-technical users

## 🐛 Bug Reports

### Before Reporting
1. Check existing issues
2. Test with the latest version
3. Gather relevant information

### Bug Report Template
```markdown
**Bug Description**
A clear description of the bug.

**Steps to Reproduce**
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Windows 10, macOS 12.0, Ubuntu 20.04]
- Python Version: [e.g., 3.9.7]
- Browser: [e.g., Chrome 95.0]

**Additional Context**
Any other relevant information.
```

## 💡 Feature Requests

### Before Requesting
1. Check if the feature already exists
2. Consider ethical implications
3. Think about user safety impact

### Feature Request Template
```markdown
**Feature Description**
Clear description of the proposed feature.

**Use Case**
Why is this feature needed? What problem does it solve?

**Ethical Considerations**
How does this feature align with our ethical guidelines?

**Implementation Ideas**
Any thoughts on how this could be implemented?

**Alternatives Considered**
Other solutions you've considered.
```

## 🔍 Pull Request Process

### PR Requirements
1. **Description**: Clear description of changes
2. **Tests**: All tests must pass
3. **Documentation**: Update relevant documentation
4. **Ethical Review**: Consider ethical implications
5. **Code Review**: At least one maintainer approval

### PR Template
```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Other (please describe)

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Ethical Considerations
- [ ] Considered impact on user safety
- [ ] Reviewed for potential bias
- [ ] Maintains privacy standards

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass locally
```

## 🏷️ Release Process

### Version Numbering
We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist
1. Update version numbers
2. Update CHANGELOG.md
3. Run full test suite
4. Create release notes
5. Tag release
6. Deploy to production

## 🆘 Getting Help

### Community Support
- **GitHub Discussions**: For general questions and ideas
- **GitHub Issues**: For bug reports and feature requests
- **Email**: For security issues or sensitive matters

### Maintainer Contact
- **Project Lead**: [maintainer@project.com]
- **Security Issues**: [security@project.com]
- **Ethical Concerns**: [ethics@project.com]

## 🙏 Recognition

Contributors will be recognized in:
- README.md acknowledgments
- Release notes
- Project documentation
- Annual contributor reports

Thank you for helping make mental health support more accessible and ethical!

---

**Remember**: This project deals with sensitive mental health data. Always prioritize user safety and well-being in your contributions.
