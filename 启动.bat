@echo off
chcp 936 >nul
setlocal
title DisSolve 一键启动

rem ===== 切到脚本所在目录（项目根目录）=====
cd /d "%~dp0"

echo ============================================================
echo    DisSolve - 水溶解度预测系统  一键启动
echo ============================================================
echo.

rem ---------- 1. 环境自检 ----------
if not exist "venv\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境: venv\Scripts\python.exe
    echo        请在项目根目录执行:  python -m venv venv
    goto :fail
)

if not exist "frontend\package.json" (
    echo [错误] 未找到前端目录: frontend\package.json
    goto :fail
)

where npm >nul 2>nul
if errorlevel 1 (
    echo [错误] 未检测到 npm，请先安装 Node.js 并加入 PATH
    goto :fail
)

if not exist "frontend\node_modules" (
    echo [提示] 首次运行，正在安装前端依赖，请稍候...
    pushd frontend
    call npm install
    popd
    if errorlevel 1 (
        echo [错误] 前端依赖安装失败，请检查网络或 npm 源
        goto :fail
    )
)

rem ---------- 2. 选空闲端口 ----------
rem 8000 常被其它本地服务占用（例如「文献评估」后端）。被占用时 uvicorn 会
rem 绑定失败并退出，前端所有 /api 请求随之 404，界面上只显示“无法连接后端服务”。
rem 这里先找一个真正空闲的端口，再让前端代理指向同一个端口。
for /f %%P in ('powershell -NoProfile -Command "$p=8000; while(Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue){$p++}; $p"') do set "API_PORT=%%P"
for /f %%P in ('powershell -NoProfile -Command "$p=3000; while(Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue){$p++}; $p"') do set "WEB_PORT=%%P"

if not "%API_PORT%"=="8000" (
    echo [提示] 端口 8000 已被其它程序占用，后端改用 %API_PORT%
)
if not "%WEB_PORT%"=="3000" (
    echo [提示] 端口 3000 已被其它程序占用，前端改用 %WEB_PORT%
)

rem ---------- 3. 启动后端 API ----------
echo [1/3] 启动后端 API    ...  http://127.0.0.1:%API_PORT%
start "DisSolve Backend" cmd /k venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port %API_PORT%

rem ---------- 4. 启动前端（代理指向同一个后端端口）----------
echo [2/3] 启动前端        ...  http://localhost:%WEB_PORT%
set "VITE_API_TARGET=http://127.0.0.1:%API_PORT%"
start "DisSolve Frontend" cmd /k "cd /d frontend && npm run dev -- --port %WEB_PORT% --strictPort"

rem ---------- 5. 等后端就绪再开浏览器 ----------
echo [3/3] 等待后端就绪...
powershell -NoProfile -Command "for($i=0;$i -lt 60;$i++){try{Invoke-WebRequest -Uri 'http://127.0.0.1:%API_PORT%/api/health' -UseBasicParsing -TimeoutSec 2 | Out-Null; exit 0}catch{Start-Sleep -Milliseconds 500}}; exit 1"
if errorlevel 1 (
    echo [警告] 后端 30 秒内没有响应，请查看 "DisSolve Backend" 窗口里的报错。
) else (
    start "" "http://localhost:%WEB_PORT%"
)

echo.
echo 启动完成。
echo   - 后端: http://127.0.0.1:%API_PORT%   窗口 "DisSolve Backend"
echo   - 前端: http://localhost:%WEB_PORT%   窗口 "DisSolve Frontend"
echo   - 关闭本窗口不会停止服务；要停止请关闭那两个命令行窗口。
echo.
pause
exit /b 0

:fail
echo.
echo 启动失败，请按上面的提示排查。
pause
exit /b 1