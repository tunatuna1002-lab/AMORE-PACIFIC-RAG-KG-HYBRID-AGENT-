# Prompt Version Management System

## Overview

The prompt version management system allows you to:
- Manage multiple versions of prompts
- Perform A/B testing with weighted selection
- Track usage and success metrics
- Persist metrics across sessions

## Basic Usage

```python
from prompts import get_prompt_manager

# Get the singleton manager
manager = get_prompt_manager()

# Register a new version
manager.register_version(
    name="insight_generation",
    version="v1",
    content="Generate insights about {topic} using {data}",
    metadata={"author": "team", "created": "2026-01-28"}
)

manager.register_version(
    name="insight_generation",
    version="v2",
    content="Analyze {topic} and provide strategic insights based on {data}",
    metadata={"author": "team", "created": "2026-01-28"}
)

# Get a specific version
prompt = manager.get_prompt(
    "insight_generation",
    version="v1",
    topic="market trends",
    data="sales data"
)

# Get the latest version (highest version number)
prompt = manager.get_prompt(
    "insight_generation",
    version="latest",
    topic="market trends",
    data="sales data"
)
```

## A/B Testing

```python
# Perform A/B testing with weighted selection
prompt, selected_version = manager.get_prompt_ab(
    "insight_generation",
    weights={"v1": 0.3, "v2": 0.7},  # 30% v1, 70% v2
    topic="market trends",
    data="sales data"
)

print(f"Selected version: {selected_version}")
print(f"Prompt: {prompt}")
```

## Tracking Success

```python
# Record when a prompt leads to a successful outcome
manager.record_success("insight_generation", "v2")

# Get metrics for all versions
metrics = manager.get_metrics("insight_generation")
# {
#   "v1": {
#     "usage_count": 10,
#     "success_count": 7,
#     "success_rate": 0.7
#   },
#   "v2": {
#     "usage_count": 20,
#     "success_count": 16,
#     "success_rate": 0.8
#   }
# }
```

## Metrics Persistence

Metrics are automatically saved to `prompts/metrics.json` and loaded on initialization:

```json
{
  "insight_generation": {
    "v1": {
      "usage_count": 10,
      "success_count": 7
    },
    "v2": {
      "usage_count": 20,
      "success_count": 16
    }
  }
}
```

## Integration with Existing Code

The version manager works alongside the existing `PromptLoader`:

```python
# Old way (still works)
from prompts import PromptLoader
prompt = PromptLoader.format("insight_generation", topic="trends")

# New way (with versioning)
from prompts import get_prompt_manager
manager = get_prompt_manager()
prompt = manager.get_prompt("insight_generation", version="v2", topic="trends")
```

## Example: Migration Path

```python
from prompts import get_prompt_manager, PromptLoader

# Initialize manager
manager = get_prompt_manager()

# Load existing prompts from files and register as v1
for prompt_name in ["insight_generation", "chatbot_system", "analysis"]:
    content = PromptLoader.get(prompt_name)
    manager.register_version(prompt_name, "v1", content)

# Now you can experiment with new versions
manager.register_version(
    "insight_generation",
    "v2",
    "Improved prompt with better instructions..."
)

# A/B test between old and new
prompt, version = manager.get_prompt_ab(
    "insight_generation",
    weights={"v1": 0.5, "v2": 0.5}
)
```

## Best Practices

1. **Semantic Versioning**: Use v1, v2, v3 for major changes
2. **Metadata**: Include creation date, author, and change notes
3. **A/B Testing**: Start with 50/50 split, then adjust based on metrics
4. **Success Tracking**: Always call `record_success()` when a prompt achieves its goal
5. **Regular Review**: Check metrics weekly to identify best-performing versions

## API Reference

### `PromptVersionManager`

- `register_version(name, version, content, metadata=None)` - Register a new prompt version
- `get_prompt(name, version="latest", **kwargs)` - Get a prompt with formatting
- `get_prompt_ab(name, weights, **kwargs)` - A/B test with weighted selection
- `record_success(name, version)` - Record a successful outcome
- `get_metrics(name)` - Get usage and success metrics

### `get_prompt_manager()`

Returns the singleton `PromptVersionManager` instance.
