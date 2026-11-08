from pathlib import Path
import zipfile
root=Path(__file__).resolve().parents[1]
out=root/'dist'/'support-ticket-router-source.zip'
out.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in root.rglob('*'):
        relative=path.relative_to(root)
        if path.is_file() and path.suffix not in {'.zip','.safetensors','.pt','.bin'} and not any(p in {'.venv','__pycache__','artifacts','.git','data','dist'} or p.endswith('.egg-info') for p in relative.parts) and path.name!='.env':
            archive.write(path,relative.as_posix())
print(out)
