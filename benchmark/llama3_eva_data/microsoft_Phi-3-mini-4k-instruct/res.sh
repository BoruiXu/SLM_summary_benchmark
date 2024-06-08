#!/bin/bash

# 定义目标字符串
target="ave"

# 获取当前目录下所有包含目标字符串的文件
files=$(ls | grep "$target")

# 检查是否找到符合条件的文件
if [ -z "$files" ]; then
  echo "当前目录下没有文件名中包含 '$target' 的文件。"
else
  echo "文件名中包含 '$target' 的文件如下："
  echo "$files"
  echo

  # 输出这些文件的内容
  for file in $files; do
    if [ -f "$file" ]; then
      echo "文件: $file 的内容如下："
      cat "$file"
      echo
    else
      echo "$file 不是一个常规文件，跳过。"
      echo
    fi
  done
fi

