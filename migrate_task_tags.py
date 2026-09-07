"""
One-off migration: promotes the "dur:"/"effort:"/"when:" note-tag syntax
(project_manager.py's _parse_note_tags) into the first-class task fields
duration_minutes/effort/when, and strips the tag lines out of notes.

Run in preview mode first (the default - no files are touched):

    python migrate_task_tags.py

Then, once the preview output looks right, apply it for real:

    python migrate_task_tags.py --apply

Safe to re-run: a task with no recognized tag lines at the top of its notes
(including one already migrated) is left untouched, since _parse_note_tags
returns no tags for it.
"""
import argparse
import json
import shutil
from datetime import datetime, timezone

import project_manager as pm


def migrate(state: dict) -> list[dict]:
    """Mutates `state` in place, promoting note-tags into first-class
    fields on every task that has any. Returns a list of per-task change
    summaries (for preview/log output) - empty list means nothing to do."""
    changes = []
    for project in state["projects"].values():
        for task in project["tasks"].values():
            tags, prose = pm._parse_note_tags(task.get("notes", ""))
            if not tags:
                continue

            before = {
                "notes": task.get("notes", ""),
                "duration_minutes": task.get("duration_minutes"),
                "effort": task.get("effort", ""),
                "when": task.get("when", ""),
            }

            if "dur" in tags:
                task["duration_minutes"] = tags["dur"]
            if "effort" in tags:
                task["effort"] = tags["effort"]
            if "when" in tags:
                task["when"] = tags["when"]
            task["notes"] = prose

            changes.append({
                "project": f"{project['short_code']} — {project['name']}",
                "task": f"{task['short_id']} {task['title']}",
                "before": before,
                "after": {
                    "notes": task["notes"],
                    "duration_minutes": task.get("duration_minutes"),
                    "effort": task.get("effort", ""),
                    "when": task.get("when", ""),
                },
            })
    return changes


def _print_change(change: dict) -> None:
    print(f"\n[{change['project']}] {change['task']}")
    b, a = change["before"], change["after"]
    if b["duration_minutes"] != a["duration_minutes"]:
        print(f"  duration_minutes: {b['duration_minutes']!r} -> {a['duration_minutes']!r}")
    if b["effort"] != a["effort"]:
        print(f"  effort:           {b['effort']!r} -> {a['effort']!r}")
    if b["when"] != a["when"]:
        print(f"  when:             {b['when']!r} -> {a['when']!r}")
    print(f"  notes:            {b['notes']!r}")
    print(f"               ->   {a['notes']!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually write the changes. Without this flag, only previews them.",
    )
    args = parser.parse_args()

    state = pm._load()
    changes = migrate(state)

    if not changes:
        print("Nothing to migrate - no task has recognized note-tags.")
        return

    for change in changes:
        _print_change(change)

    print(f"\n{len(changes)} task(s) would be migrated.")

    if not args.apply:
        print("\nPreview only - no files were changed. Re-run with --apply to write these changes.")
        return

    backup_path = pm.STATE_FILE.with_name(
        f"{pm.STATE_FILE.stem}.bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}{pm.STATE_FILE.suffix}"
    )
    shutil.copy2(pm.STATE_FILE, backup_path)
    print(f"\nBacked up {pm.STATE_FILE} -> {backup_path}")

    pm._save(state)
    print(f"Applied {len(changes)} change(s) to {pm.STATE_FILE}.")


if __name__ == "__main__":
    main()
