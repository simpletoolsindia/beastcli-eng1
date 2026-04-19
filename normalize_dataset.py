#!/usr/bin/env python3
"""
normalize_dataset.py — Fixes the schema issues in existing Beast CLI datasets.

Issues fixed:
1. tool messages use {"status":"success"} instead of {"type":"tool_result",...}
2. assistant messages with text content (not JSON tool_call/final_answer)
3. missing type fields in JSON content
4. tool_call_id not present in tool messages

Usage:
    python normalize_dataset.py input.jsonl output.jsonl [--dry-run]
"""

import json
import sys
import argparse
import uuid
from pathlib import Path


def parse_assistant_content(content: str | None) -> dict | None:
    """Try to parse assistant content as JSON. Return None if plain text."""
    if not content:
        return None
    content = content.strip()
    # Skip obvious plain text
    text_indicators = ['Done.', 'All done.', 'Completed.', 'Finished.', 'Processing...',
                       'Working on it...', 'Ejecutando...', 'Operacion']
    if any(content.startswith(t) for t in text_indicators):
        return None
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and 'type' in obj:
            return obj
        return None  # JSON but not a tool_call or final_answer
    except (json.JSONDecodeError, TypeError):
        return None


def fix_tool_result(content: str, tool_call_id: str | None = None) -> str:
    """Convert status-based tool response to type-based tool_result."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        # Malformed JSON — try to extract what we can
        return json.dumps({
            "type": "tool_result",
            "tool_call_id": tool_call_id or f"call_{uuid.uuid4().hex[:8]}",
            "output": content[:500],
        })

    # Already has type field
    if 'type' in data:
        return content

    # Convert from status-based format
    output_data = {k: v for k, v in data.items() if k != 'status'}
    return json.dumps({
        "type": "tool_result",
        "tool_call_id": tool_call_id or output_data.get('tool_call_id', f"call_{uuid.uuid4().hex[:8]}"),
        "output": json.dumps(output_data),
    }, ensure_ascii=False)


def fix_assistant_message(content: str, sample_idx: int, msg_idx: int, msgs: list) -> str | None:
    """Ensure assistant message content is valid JSON: tool_call or final_answer."""
    if not content:
        return None

    content = content.strip()
    if not content:
        return None

    # Already valid JSON with type
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and obj.get('type') in ('tool_call', 'final_answer'):
            return content
    except (json.JSONDecodeError, TypeError):
        pass

    # Plain text — convert to final_answer (the final response before tool)
    # This happens when the assistant response is just text (like "All done.")
    return json.dumps({
        "type": "final_answer",
        "content": content[:500],  # truncate long text
    }, ensure_ascii=False)


def normalize_sample(sample: dict, idx: int) -> dict | None:
    """Normalize a single dataset sample to Unsloth-compatible format."""
    messages = sample.get('messages', [])
    if not messages:
        return None

    normalized = {"messages": []}

    prev_tool_id = None
    for msg_idx, msg in enumerate(messages):
        role = msg.get('role', '')
        content = msg.get('content')
        tool_call_id = msg.get('tool_call_id')

        if role == 'system':
            normalized['messages'].append({
                "role": "system",
                "content": content or "",
            })

        elif role == 'user':
            normalized['messages'].append({
                "role": "user",
                "content": content or "",
            })

        elif role == 'assistant':
            if content:
                parsed = parse_assistant_content(content)
                if parsed and parsed.get('type') == 'tool_call':
                    # Valid tool_call JSON string
                    normalized['messages'].append({
                        "role": "assistant",
                        "content": content,
                    })
                elif parsed and parsed.get('type') == 'final_answer':
                    normalized['messages'].append({
                        "role": "assistant",
                        "content": content,
                    })
                else:
                    # Plain text — convert to final_answer
                    fixed = fix_assistant_message(content, idx, msg_idx, messages)
                    if fixed:
                        normalized['messages'].append({
                            "role": "assistant",
                            "content": fixed,
                        })
            elif msg.get('tool_calls'):
                # tool_calls format — convert to JSON string
                tc = msg['tool_calls'][0] if msg['tool_calls'] else {}
                func = tc.get('function', {})
                tool_call_json = json.dumps({
                    "type": "tool_call",
                    "tool_name": func.get('name', 'unknown'),
                    "arguments": json.loads(func.get('arguments', '{}')),
                }, ensure_ascii=False)
                normalized['messages'].append({
                    "role": "assistant",
                    "content": tool_call_json,
                })

        elif role == 'tool':
            if content:
                fixed = fix_tool_result(content, tool_call_id)
                normalized['messages'].append({
                    "role": "tool",
                    "content": fixed,
                    "tool_call_id": tool_call_id or f"call_{uuid.uuid4().hex[:8]}",
                })
            else:
                # Empty content — add empty tool_result
                normalized['messages'].append({
                    "role": "tool",
                    "content": json.dumps({
                        "type": "tool_result",
                        "tool_call_id": tool_call_id or f"call_{uuid.uuid4().hex[:8]}",
                        "output": "",
                    }),
                    "tool_call_id": tool_call_id or f"call_{uuid.uuid4().hex[:8]}",
                })

    # Ensure last message is final_answer assistant
    if normalized['messages']:
        last = normalized['messages'][-1]
        if last['role'] != 'assistant':
            # No assistant final answer — add one
            normalized['messages'].append({
                "role": "assistant",
                "content": json.dumps({
                    "type": "final_answer",
                    "content": "Task completed.",
                }, ensure_ascii=False),
            })
        else:
            # Check if it's valid final_answer JSON
            try:
                obj = json.loads(last['content'])
                if obj.get('type') != 'final_answer':
                    # Not a final_answer — add proper final_answer
                    normalized['messages'].append({
                        "role": "assistant",
                        "content": json.dumps({
                            "type": "final_answer",
                            "content": "Done.",
                        }, ensure_ascii=False),
                    })
            except (json.JSONDecodeError, TypeError):
                # Not valid JSON — convert to final_answer
                last['content'] = json.dumps({
                    "type": "final_answer",
                    "content": last['content'][:200],
                }, ensure_ascii=False)

    # Carry over metadata
    if 'localization' in sample:
        normalized['localization'] = sample['localization']
    if 'metadata' in sample:
        normalized['metadata'] = sample['metadata']
    if 'tools' in sample:
        normalized['tools'] = sample['tools']

    return normalized


def audit_file(path: str) -> dict:
    """Audit a JSONL file and report issues."""
    issues = {
        'total': 0, 'parsed': 0, 'errors': 0,
        'non_json_assistant': 0, 'wrong_tool_schema': 0,
        'no_final_answer': 0, 'empty_tool_result': 0,
    }

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            issues['total'] += 1
            try:
                sample = json.loads(line)
                issues['parsed'] += 1
            except:
                issues['errors'] += 1
                continue

            msgs = sample.get('messages', [])
            for m in msgs:
                if m['role'] == 'assistant':
                    c = m.get('content', '')
                    try:
                        obj = json.loads(c)
                        if not isinstance(obj, dict) or 'type' not in obj:
                            issues['non_json_assistant'] += 1
                    except:
                        issues['non_json_assistant'] += 1
                elif m['role'] == 'tool':
                    c = m.get('content', '')
                    try:
                        obj = json.loads(c)
                        if obj.get('type') != 'tool_result':
                            issues['wrong_tool_schema'] += 1
                    except:
                        issues['wrong_tool_schema'] += 1

            # Check final answer
            has_final = False
            if msgs and msgs[-1]['role'] == 'assistant':
                try:
                    obj = json.loads(msgs[-1]['content'])
                    if obj.get('type') == 'final_answer':
                        has_final = True
                except:
                    pass
            if not has_final:
                issues['no_final_answer'] += 1

    return issues


def main():
    parser = argparse.ArgumentParser(description='Normalize Beast CLI dataset to Unsloth format')
    parser.add_argument('input', help='Input JSONL file')
    parser.add_argument('output', help='Output JSONL file')
    parser.add_argument('--dry-run', action='store_true', help='Audit only, no writes')
    parser.add_argument('--audit', action='store_true', help='Audit only')
    args = parser.parse_args()

    if args.audit:
        print(f"Auditing {args.input}...")
        issues = audit_file(args.input)
        print(f"Total: {issues['total']}, Parsed: {issues['parsed']}, Errors: {issues['errors']}")
        print(f"Non-JSON assistant: {issues['non_json_assistant']}")
        print(f"Wrong tool schema: {issues['wrong_tool_schema']}")
        print(f"No final_answer: {issues['no_final_answer']}")
        return

    if args.dry_run:
        print(f"Auditing {args.input} (dry run — no writes)...")
        issues = audit_file(args.input)
        print(f"Would fix: {issues['total']} samples")
        return

    print(f"Normalizing {args.input} -> {args.output}...")
    fixed = 0
    skipped = 0

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, 'r', encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(fin):
            try:
                sample = json.loads(line)
                normalized = normalize_sample(sample, i)
                if normalized:
                    fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    fixed += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"  Error on line {i}: {e}")
                skipped += 1

            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1} samples...")

    print(f"\nDone. Fixed: {fixed}, Skipped: {skipped}")
    print(f"Output: {args.output}")


if __name__ == '__main__':
    main()