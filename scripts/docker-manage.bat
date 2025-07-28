@echo off
REM Docker management script for NIDS on Windows

setlocal enabledelayedexpansion

set ENVIRONMENT=development
set ACTION=build
set PUSH_IMAGES=false
set REGISTRY=

:parse_args
if "%~1"=="" goto :main
if "%~1"=="-e" (
    set ENVIRONMENT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--environment" (
    set ENVIRONMENT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-a" (
    set ACTION=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--action" (
    set ACTION=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-p" (
    set PUSH_IMAGES=true
    shift
    goto :parse_args
)
if "%~1"=="--push" (
    set PUSH_IMAGES=true
    shift
    goto :parse_args
)
if "%~1"=="-r" (
    set REGISTRY=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--registry" (
    set REGISTRY=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-h" goto :help
if "%~1"=="--help" goto :help
echo Unknown option: %~1
goto :help

:help
echo Usage: %0 [OPTIONS]
echo Options:
echo   -e, --environment    Environment (development^|production) [default: development]
echo   -a, --action         Action (build^|up^|down^|restart^|logs) [default: build]
echo   -p, --push          Push images to registry
echo   -r, --registry      Docker registry URL
echo   -h, --help          Show this help message
exit /b 0

:main
echo Building NIDS Docker setup for %ENVIRONMENT% environment

if "%ACTION%"=="build" goto :build
if "%ACTION%"=="up" goto :up
if "%ACTION%"=="down" goto :down
if "%ACTION%"=="restart" goto :restart
if "%ACTION%"=="logs" goto :logs
echo Unknown action: %ACTION%
goto :help

:build
echo Building Docker images...
docker build --target base -t nids:base .
docker build --target api -t nids:api .
docker build --target dashboard -t nids:dashboard .
docker build --target training -t nids:training .
docker build --target capture -t nids:capture .

if not "%REGISTRY%"=="" (
    echo Tagging images with registry %REGISTRY%...
    docker tag nids:api %REGISTRY%/nids:api
    docker tag nids:dashboard %REGISTRY%/nids:dashboard
    docker tag nids:training %REGISTRY%/nids:training
    docker tag nids:capture %REGISTRY%/nids:capture
)

if "%PUSH_IMAGES%"=="true" (
    if not "%REGISTRY%"=="" (
        echo Pushing images to registry...
        docker push %REGISTRY%/nids:api
        docker push %REGISTRY%/nids:dashboard
        docker push %REGISTRY%/nids:training
        docker push %REGISTRY%/nids:capture
    )
)
goto :end

:up
if "%ENVIRONMENT%"=="production" (
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
) else (
    docker-compose up -d
)
goto :end

:down
docker-compose down
goto :end

:restart
docker-compose restart
goto :end

:logs
docker-compose logs -f
goto :end

:end
echo Docker management completed!
docker images | findstr nids