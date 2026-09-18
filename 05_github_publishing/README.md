# GitHub publishing

The repository-level publishing helper is kept here so it is never included in the Shiny deployment package.

```bash
python 05_github_publishing/publish_github.py --dry-run
```

Use explicit repository-relative paths when publishing changes. The helper validates the selected scope before committing or pushing.
