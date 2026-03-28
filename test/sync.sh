#!/bin/bash

# --- 配置区域 ---
# 建议在这里填入你的信息，或者直接在终端运行 git config --global
GIT_USER_NAME="HerbertLx"
GIT_USER_EMAIL="122090877@link.cuhk.edu.cn"

# 1. 自动配置身份（防止在服务器上以 root 身份乱提交）
git config user.name "$GIT_USER_NAME"
git config user.email "$GIT_USER_EMAIL"

# 2. 检查是否有文件变动
if [ -z "$(git status --porcelain)" ]; then
    echo "✨ 没有发现任何改动，无需同步。"
    exit 0
fi

echo "------------------------------------------------"
echo "📂 检测到以下文件发生了变动："
git status -s
echo "------------------------------------------------"

# 3. 获取描述
echo "🚀 请输入本次提交的描述 (Description):"
read desc

if [ -z "$desc" ]; then
  file_count=$(git status --porcelain | wc -l)
  desc="Update $file_count files at $(date +'%Y-%m-%d %H:%M:%S')"
fi

# 4. 执行 Git 三部曲
echo "正在暂存文件..."
git add .

echo "正在提交为: $GIT_USER_NAME <$GIT_USER_EMAIL>"
git commit -m "$desc"

# 自动获取当前所在的分支名，避免硬编码 main 导致推送失败
current_branch=$(git branch --show-current)

echo "正在推送到 GitHub 的 $current_branch 分支..."
git push origin "$current_branch"

echo "✅ 同步完成！"