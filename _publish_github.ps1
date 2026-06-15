Set-Location $PSScriptRoot
$repoUrl = "https://github.com/shonorovv/nauch_sfu.git"

Write-Host "=== Публикация диплома на GitHub ===" -ForegroundColor Cyan
Write-Host "Репозиторий: $repoUrl" -ForegroundColor Gray

# Удаляем мусорные файлы
foreach ($f in @("_install_cupy.py", "_run_install.bat", "_install_result.txt",
                  "analysis\_test_write.py", "test_w.py")) {
    if (Test-Path $f) {
        Remove-Item $f -Force
        Write-Host "Удалён: $f" -ForegroundColor Yellow
    }
}

# Удаляем незаконченный .git если он есть (без объектов/коммитов)
# и создаём новый, чистый репозиторий
if (Test-Path ".git") {
    Remove-Item -Recurse -Force ".git"
    Write-Host "Удалён старый .git" -ForegroundColor Yellow
}
git init -b main
Write-Host "git init выполнен" -ForegroundColor Green

# Устанавливаем remote
$existingRemotes = git remote 2>&1
if ($existingRemotes -match "origin") {
    git remote set-url origin $repoUrl
} else {
    git remote add origin $repoUrl
}
Write-Host "Remote установлен: $repoUrl" -ForegroundColor Green

# Git identity (нужно для коммита)
git config user.email "shonorov.design@gmail.com"
git config user.name "shonorovv"

# Добавляем все файлы
Write-Host "`nДобавляем файлы..." -ForegroundColor Yellow
git add -A

# Статус
git status --short

# Коммит
$msg = "diplom: Fokker-Planck GPU (CuPy 14.1), stride memory fix, backend.py"
git commit -m $msg
$commitCode = $LASTEXITCODE
if ($commitCode -eq 0) {
    Write-Host "Коммит создан" -ForegroundColor Green
} else {
    Write-Host "Нечего коммитить (всё уже актуально)" -ForegroundColor Gray
}

# Force push на main
Write-Host "`nПушим на GitHub (force)..." -ForegroundColor Yellow
git branch -M main
git push --force origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== Готово ===" -ForegroundColor Green
    Write-Host "https://github.com/shonorovv/nauch_sfu" -ForegroundColor Cyan
} else {
    Write-Host "`nОшибка push. Возможно нужна аутентификация." -ForegroundColor Red
    Write-Host "Открой эту ссылку и создай PAT (токен):" -ForegroundColor Yellow
    Write-Host "https://github.com/settings/tokens/new" -ForegroundColor Cyan
    Write-Host "Потом выполни вручную:" -ForegroundColor Yellow
    Write-Host "git remote set-url origin https://<TOKEN>@github.com/shonorovv/nauch_sfu.git" -ForegroundColor Cyan
    Write-Host "git push --force origin main" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Нажми Enter для выхода..."
Read-Host
