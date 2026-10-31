# Recorded eval questions

Voice mode (`python eval/eval_runner.py --mode voice`) uses a recording from this folder
when a turn in `eval/study_eval_set.json` points to one and the file exists. Every other
turn falls back to the question synthesized with TTS, so the eval runs with or without
these files. The report shows how many voice turns used recorded vs synthetic audio.

Only add recordings of your own voice (or synthetic audio). No third-party audio.

## What to record

Say each sentence once, naturally, in a quiet room. 2–5 seconds each is plenty.
Any of `.m4a`, `.wav`, `.mp3`, `.ogg`, `.webm` or `.flac` works: keep the file name and
change only the extension if needed (the runner looks for the same name with any of them).

| File | Language | Say exactly |
|---|---|---|
| `en_calc.m4a` | English | What is 37 times 48? |
| `hi_percent.m4a` | Hindi | 480 का 25 प्रतिशत कितना है? *(char sau assi ka pachchees pratishat kitna hai?)* |
| `hinglish_convert.m4a` | Hinglish | 5 kilometer ko miles mein convert karo |
| `ta_convert.m4a` | Tamil | 25 கிலோகிராம் எத்தனை பவுண்டு? *(irupathi ainthu kilogram ethanai pound?)* |
| `tanglish_formula.m4a` | Tanglish | Ohm's law formula enna? |
| `en_followup_1.m4a` | English (follow-up, turn 1) | 25 kg in pounds? |
| `en_followup_2.m4a` | English (follow-up, turn 2) | and in grams? |

On a Mac, Voice Memos saves `.m4a`: record, then Share → Save to Files into this folder
and rename. QuickTime Player → File → New Audio Recording also works.

Then run:

```bash
python eval/eval_runner.py --mode voice --cases calc-en,percent-hi,convert-hinglish,convert-ta,formula-tanglish,followup-en
```

## samples/question.m4a

`samples/` is git-ignored (it holds personal recordings). For the CLI quick start, record
any supported question, for example "What is 37 times 48?", and save it as
`samples/question.m4a`, then run `python cli.py --audio samples/question.m4a`.
