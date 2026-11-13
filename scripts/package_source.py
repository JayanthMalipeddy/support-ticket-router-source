from pathlib import Path
import zipfile
root=Path(__file__).resolve().parents[1]
out=root/'dist'/'support-ticket-router-source.zip'
out.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
    # Explicit source roots prevent submission scripts, credentials and private
    # evidence from being swept into a Colab upload or future source release.
    paths = [root / name for name in ('README.md', 'pyproject.toml', '.gitignore', '.gitattributes')]
    for folder in ('docs', 'examples', 'notebooks', 'scripts', 'tests', 'ticket_router', 'records'):
        paths.extend((root / folder).rglob('*'))
    for path in paths:
        relative=path.relative_to(root)
        if path.is_file() and (path.name in {'.gitignore', '.gitattributes'} or path.suffix in {'.md', '.py', '.toml', '.json', '.jsonl', '.yaml', '.yml', '.txt', '.ipynb', '.ps1'}) and not any(p in {'.venv','__pycache__','artifacts','.git','data','dist','submission'} or p.endswith('.egg-info') for p in relative.parts) and not path.name.startswith(('.env', 'credentials', 'secrets')) and not any(term in path.name.lower() for term in ('demo-script', 'demo_script', 'service-account')):
            archive.write(path,relative.as_posix())
print(out)
