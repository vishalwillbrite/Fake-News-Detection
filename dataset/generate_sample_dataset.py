"""
generate_sample_dataset.py
---------------------------
Generates a synthetic Fake.csv and True.csv sample dataset so the full
ML pipeline (preprocessing -> TF-IDF -> Logistic Regression) can be
trained and demoed end-to-end without an internet connection.

IMPORTANT (read this before submitting as a final project):
This script produces a SMALL, TEMPLATE-BASED sample dataset purely so
that train_model.py has something to run against out of the box.
For a real submission, replace dataset/Fake.csv and dataset/True.csv
with the actual "Fake and Real News Dataset" (ISOT dataset) from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

The real dataset contains 44,898 labeled articles (23,481 fake / 21,417
real) and is what the README's reported statistics assume.
"""

import csv
import random

random.seed(42)

REAL_SUBJECTS = ["politicsNews", "worldnews"]
FAKE_SUBJECTS = ["News", "politics", "Government News", "left-news"]

REAL_TEMPLATES = [
    "The {body} announced on {day} that it will {action} following weeks "
    "of negotiations between officials in {place}. Analysts said the "
    "decision reflects a broader shift in policy after months of debate "
    "in the legislature. Spokespeople confirmed the plan will be "
    "reviewed again next quarter, with a formal report expected from "
    "the committee overseeing implementation.",
    "Officials in {place} confirmed on {day} that the {body} reached an "
    "agreement to {action}, according to a statement released to "
    "reporters. The agreement, which took several months to finalize, "
    "was welcomed by trade groups and criticized by some opposition "
    "lawmakers who argued the timeline was too aggressive.",
    "A new report released by the {body} on {day} shows that efforts to "
    "{action} have had a measurable impact in {place}. Researchers who "
    "compiled the data said further study is needed before drawing firm "
    "conclusions, but early indicators are consistent with projections "
    "made earlier in the year.",
    "The {body} said in a press briefing on {day} that plans to "
    "{action} in {place} are moving forward as scheduled. The "
    "announcement came after a series of closed-door meetings with "
    "regional representatives, several of whom later spoke to reporters "
    "on condition of anonymity.",
]

FAKE_TEMPLATES = [
    "You won't believe what {body} is secretly hiding about {place}! "
    "Insiders reveal that officials plan to {action} while the "
    "mainstream media refuses to report the shocking truth. Share this "
    "before it gets taken down!!!",
    "BREAKING: {body} caught in massive cover-up to {action} in "
    "{place}, according to anonymous sources close to the situation. "
    "Experts who spoke to us say this changes everything, and yet "
    "nobody in the corporate press wants to talk about it.",
    "Leaked documents allegedly prove that {body} has been planning to "
    "{action} in {place} for years without telling the public. Critics "
    "say this is the biggest scandal of the decade and demand answers "
    "immediately from every official involved.",
    "Shocking new claims suggest {body} is secretly working to "
    "{action} in {place}, sparking outrage among viewers who saw the "
    "viral video first. Multiple unverified accounts insist the story "
    "is being suppressed by big tech companies.",
]

BODIES = ["the White House", "the Senate committee", "the state department",
          "the central bank", "the trade ministry", "the governor's office",
          "the United Nations panel", "the defense department"]
ACTIONS = ["raise import tariffs", "expand healthcare subsidies",
           "reform the tax code", "increase military funding",
           "tighten border regulations", "cut corporate tax rates",
           "launch a new infrastructure program", "revise trade agreements"]
PLACES = ["Washington", "California", "the Midwest", "New York",
          "the European Union", "the Gulf region", "Texas", "the Pacific Northwest"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

REAL_TITLES = [
    "Lawmakers reach deal on {action}",
    "{place} officials confirm new policy on trade",
    "Report details progress on {action}",
    "Committee to review {action} plan next month",
]
FAKE_TITLES = [
    "SHOCKING: What they don't want you to know about {place}",
    "You won't believe this secret plan for {place}",
    "Leaked: The truth about {action} they're hiding",
    "BREAKING: Insider exposes plot to {action}",
]


def make_articles(n, templates, titles, subjects, label_name):
    rows = []
    for i in range(n):
        body_word = random.choice(BODIES)
        action = random.choice(ACTIONS)
        place = random.choice(PLACES)
        day = random.choice(DAYS)
        template = random.choice(templates)
        title_template = random.choice(titles)
        text = template.format(body=body_word, action=action, place=place, day=day)
        # add a bit of length/variety by repeating a second paragraph
        text2 = random.choice(templates).format(
            body=random.choice(BODIES), action=random.choice(ACTIONS),
            place=random.choice(PLACES), day=random.choice(DAYS))
        full_text = text + " " + text2
        title = title_template.format(action=action, place=place)
        subject = random.choice(subjects)
        date = f"{random.choice(DAYS)}, {random.randint(1,28)} {random.choice(['January','February','March','April','May','June','July','August','September','October','November','December'])}, {random.randint(2015,2018)}"
        rows.append([title, full_text, subject, date])
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "text", "subject", "date"])
        writer.writerows(rows)


if __name__ == "__main__":
    n_per_class = 600  # small sample; swap in real Kaggle CSVs for production
    fake_rows = make_articles(n_per_class, FAKE_TEMPLATES, FAKE_TITLES, FAKE_SUBJECTS, "fake")
    real_rows = make_articles(n_per_class, REAL_TEMPLATES, REAL_TITLES, REAL_SUBJECTS, "real")

    write_csv("Fake.csv", fake_rows)
    write_csv("True.csv", real_rows)

    print(f"Generated Fake.csv with {len(fake_rows)} rows")
    print(f"Generated True.csv with {len(real_rows)} rows")
