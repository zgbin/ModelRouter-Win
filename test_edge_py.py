import subprocess
import json
import os

# 确保在 /tmp 有 marked
os.chdir('/tmp')

# 写入文件给 node
with open('/tmp/edge1.md', 'w', encoding='utf-8') as f:
    f.write("""| 名称 | 描述 | 状态 |
|------|------|------|
| 张三 | 备注: 这个项目 | 完成 |
| 李四 | 描述含<br>换行 | 进行中 |
| 王五 | 含<HTML>标签和emoji | 未开始 |
| 赵六 | 包含\\|竖线符号测试 | 进行中 |""")

# 调用 node 执行 marked
result = subprocess.run(
    ['node', '-e', '''
const marked = require("marked");
const fs = require("fs");
const md = fs.readFileSync("/tmp/edge1.md", "utf8");
console.log(marked.parse(md));
'''],
    capture_output=True, text=True, encoding='utf-8'
)

print("=== 模型返回内容 (JSON解析后) ===")
with open('/tmp/edge1.md', 'r', encoding='utf-8') as f:
    print(f.read())
print("\n=== marked 解析后 HTML ===")
print(result.stdout)
if result.stderr:
    print("\n=== stderr ===")
    print(result.stderr)

# 测试其他边界情况
edge_cases = [
    ('cell_br', '| 标题 | 内容 |\n|------|------|\n| A | 第一行<br>第二行<br>第三行 |'),
    ('html_entity', '| 字段 | 值 |\n|------|------|\n| XML | <tag>内容</tag> |'),
    ('pipe_escape', '| 字段 | 值 |\n|------|------|\n| test | a\\|b\\|c |'),
    ('long_text', '| 列1 | 列2 |\n|-----|-----|\n| 长 | ' + 'x'*200 + ' |'),
    ('emoji', '| 名字 | 状态 |\n|------|------|\n| 🚀火箭 | ✅完成 |\n| 🐛Bug | ❌失败 |'),
    ('xss_test', '| 字段 | 值 |\n|------|------|\n| XSS | <script>alert(1)</script> |'),
]

for name, md in edge_cases:
    with open(f'/tmp/edge_{name}.md', 'w', encoding='utf-8') as f:
        f.write(md)
    r = subprocess.run(
        ['node', '-e', f'''
const marked = require("marked");
const fs = require("fs");
const md = fs.readFileSync("/tmp/edge_{name}.md", "utf8");
console.log(marked.parse(md));
'''],
        capture_output=True, text=True, encoding='utf-8'
    )
    print(f"\n=== {name} ===")
    print(r.stdout)
    if r.stderr:
        print(f"stderr: {r.stderr}")