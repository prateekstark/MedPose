# Patient data policy

These rules apply to any recording of a real patient, and to anything derived from one that could identify them.

## Rules
1. **Store it under `patient_data/`, on an encrypted disk.** Git ignores that folder, along with common video formats, pickles and C3D files.
2. **Process it locally or on a machine approved for this data.** Don't use the hosted OpenCap service; its terms allow the development team to view submitted video and use it for algorithm development.
3. **AI assistants never see it.** That covers Claude, ChatGPT and similar tools: no videos, frames, faces, names, dates of birth or medical records. Develop with public data or synthetic clips, and share only de-identified summary numbers when asking for help.
4. **Keep the person's usual supports while recording.** Never remove a walker, brace or belt to help tracking. Their physio or doctor decides what's safe.
5. **Results are prompts for clinicians.** MedPose isn't clinically validated, and its output isn't a diagnosis or a treatment decision.
6. **Publishing needs approval first.** Using a patient's recordings in a paper or dataset needs ethics approval and consent. For a child, that means parental consent and the child's own assent. Get these before recording for that purpose.

## How this is enforced, and what isn't
| Protection | What it covers | Gap |
|---|---|---|
| `.gitignore` | Keeps patient folders and common video formats out of git | `git add -f` overrides it, and a video in another format wouldn't match |
| `.claude/settings.json` deny rules | Blocks Claude's file-reading tools on `patient_data/`, `normalized/`, `opencap-core/Data/` and `opencap-monocular/results/` | Also blocked a shell `cp` of `opencap-core/Data/` in testing, but don't assume it catches every command. Scripts Claude starts can read files themselves, so run patient pipelines from your own terminal |
| `tests/test_data_policy.py` | Fails if the ignore or deny rules are removed | Only runs when the tests do |

**Core and Monocular write working copies of the video into `opencap-core/Data/` and `opencap-monocular/results/`,** so treat both folders as patient data once real recordings have been processed.
