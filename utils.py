import os
import markdown
from typing import List, Dict
import re

def read_markdown_file(file_path: str) -> str:
    """Read and return the content of a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_metadata(content: str) -> Dict[str, str]:
    """Extract YAML frontmatter from markdown content."""
    metadata = {}
    yaml_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)
    match = yaml_pattern.match(content)
    
    if match:
        yaml_content = match.group(1)
        for line in yaml_content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
    
    return metadata

def markdown_to_text(content: str) -> str:
    """Convert markdown content to plain text."""
    # Remove YAML frontmatter
    content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', content, flags=re.DOTALL)
    
    # Convert markdown to HTML and then strip HTML tags
    html = markdown.markdown(content)
    text = re.sub(r'<[^>]+>', '', html)
    return text.strip()

def get_file_id(file_path: str) -> str:
    """Generate a unique identifier for a file."""
    return os.path.splitext(os.path.basename(file_path))[0]

def is_markdown_file(file_path: str) -> bool:
    """Check if a file is a markdown file."""
    return file_path.lower().endswith(('.md', '.markdown'))

def get_all_markdown_files(directory: str) -> List[str]:
    """Get all markdown files in a directory and its subdirectories."""
    markdown_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if is_markdown_file(file):
                markdown_files.append(os.path.join(root, file))
    return markdown_files 