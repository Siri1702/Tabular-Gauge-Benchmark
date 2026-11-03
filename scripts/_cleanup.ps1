$src = 'paper1'
$dst = '_archive/paper1_v1_2026-08-19'
New-Item -ItemType Directory -Force -Path $dst | Out-Null

# Move every immediate file of paper1/ (not directories) into the archive
Get-ChildItem -Path $src -File | Move-Item -Destination $dst -Force

# Move each subdirectory in paper1/ except v2/ into the archive
Get-ChildItem -Path $src -Directory | Where-Object { $_.Name -ne 'v2' } | ForEach-Object {
    $newPath = Join-Path $dst $_.Name
    Move-Item -Path $_.FullName -Destination $newPath -Force
}

Write-Host 'paper1/ contents after cleanup:'
Get-ChildItem paper1 | Select-Object Name
