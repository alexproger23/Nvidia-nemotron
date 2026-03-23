Используй библиотеки где есть возможность.
НЕ пытайся сохранять ЛЕГАСИ код.
Веди документацию, но не плоди постоянно, иногда сокращай или делай ее рефакторинг. Документацию можно делать в папке docs, в .md файлах
README.md Не заполняется для тебя и как документация. А как небольшое описание в дальнейшем для репозитория

---
Инструкции по коммитам
One commit = one logical change (feature / fix / refactor). Keep it reviewable
Never mix: formatting-only changes, dependency bumps, and behavior changes in the same commit.
If a task touches multiple areas, split into a stack:
prep/refactor (no behavior change)
main change
follow-ups (docs, cleanup)
Commit messages (Conventional Commits):
feat: ...
fix: ...
refactor: ...
docs: ...
test: ...
chore: ...
Prefer atomic commits that leave the repo in a passing state (tests green) after each commit.