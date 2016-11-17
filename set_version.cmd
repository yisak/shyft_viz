@echo off

git rev-list --count HEAD > VERSION

set /p minor=<VERSION

echo 1.0.%minor% > VERSION

echo ##teamcity[buildNumber '1.0.%minor%']
