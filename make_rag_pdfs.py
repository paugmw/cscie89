"""Generate four small PDFs on very different subjects for the RAG exercise.

Layout produced:
    data/
        current_affairs/
            politics_city_council_reform.pdf
            sport_harbor_marathon.pdf
        lifestyle/
            cooking_saffron_paella.pdf
            travel_azores_itinerary.pdf

Each document contains a few distinctive facts (names, numbers, places) so that
a question about one subject can only be answered from one file.

Run:  python make_rag_pdfs.py
"""
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

DATA_DIR = Path("data")

DOCS = {
    "current_affairs/politics_city_council_reform.pdf": {
        "title": "Politics: The Riverton City Council Reform of 2025",
        "subject": "politics",
        "sections": [
            ("Background",
             "The fictional city of Riverton, population 412,000, was governed for "
             "forty years by a nine-member city council elected at large. Critics argued "
             "that the at-large system allowed the wealthier northern neighborhoods to "
             "dominate every election, while the industrial south side had not elected a "
             "council member since 1998. Turnout in municipal elections had fallen to "
             "just 19 percent, one of the lowest figures among comparable cities."),
            ("The Reform Proposal",
             "In March 2025 Mayor Elena Castillo introduced Proposition 7, which replaced "
             "the at-large system with eleven geographic districts and added two at-large "
             "seats, for a total council of thirteen members. The proposal also introduced "
             "ranked-choice voting, limited council members to three consecutive four-year "
             "terms, and created an independent redistricting commission of seven citizens "
             "selected by lottery from a pool of qualified volunteers."),
            ("The Campaign",
             "The Yes on 7 coalition was led by the League of Riverton Neighborhoods and "
             "the regional labor federation. Opponents, organized as Keep Riverton United, "
             "argued that districts would encourage parochial politics and that "
             "ranked-choice voting was too confusing for voters. Total campaign spending "
             "reached 2.3 million dollars, a record for a municipal ballot measure in the "
             "state. Three public debates were held at the Riverton Public Library."),
            ("The Result",
             "On November 4, 2025, Proposition 7 passed with 58.4 percent of the vote. "
             "Turnout rose to 41 percent, more than double the previous municipal "
             "election. The first district-based elections are scheduled for November "
             "2027, and the redistricting commission must publish its draft map by "
             "June 1, 2026, followed by a sixty-day public comment period."),
            ("Analysis",
             "Political scientists at Riverton State University noted that the reform "
             "follows a national trend toward district elections in mid-sized cities. "
             "Professor Daniel Okafor predicted that the south side would gain at least "
             "two seats and that the influence of developer-funded political action "
             "committees would decline. Whether ranked-choice voting will increase "
             "civility in campaigns remains, in his words, an open empirical question."),
        ],
    },
    "current_affairs/sport_harbor_marathon.pdf": {
        "title": "Sport: The 2025 Harbor City Marathon",
        "subject": "sport",
        "sections": [
            ("The Race",
             "The 31st Harbor City Marathon took place on Sunday, October 12, 2025, with "
             "18,600 registered runners from 64 countries. The course starts at the "
             "Lighthouse Pier, loops twice around Bayfront Park, crosses the Old Iron "
             "Bridge at kilometer 28, and finishes in front of the Maritime Museum. "
             "Race-day temperature at the start was a cool 11 degrees Celsius with a "
             "light westerly wind, close to ideal conditions for fast times."),
            ("Men's Race",
             "The men's race was won by Tomas Varga in 2 hours 6 minutes 41 seconds, a "
             "new course record that broke the previous mark by 52 seconds. Varga broke "
             "away from the lead pack at kilometer 35 on the long climb after the Old "
             "Iron Bridge. Second place went to Samuel Kiprono in 2:07:10 and third to "
             "the debutant Lucas Ferreira in 2:08:03."),
            ("Women's Race",
             "In the women's race, Amina Haddad defended her title in 2 hours 21 minutes "
             "15 seconds. She ran most of the race alone after the halfway point, which "
             "she reached in 70 minutes 20 seconds. It was her third victory in Harbor "
             "City, making her the most successful athlete in the event's history."),
            ("Wheelchair and Mass Participation",
             "The wheelchair race was won by Jonas Lindqvist in 1:29:58. Among amateur "
             "runners, the median finishing time was 4 hours 18 minutes, and 97.2 percent "
             "of starters reached the finish line. The race raised 1.4 million dollars "
             "for the children's hospital charity partner."),
            ("Training Notes",
             "Race director Maria Olsen recommended that first-time marathoners build "
             "up to a peak week of 55 to 70 kilometers, include one long run of 30 to 32 "
             "kilometers about three weeks before race day, and practice drinking at the "
             "aid stations, which are placed every 5 kilometers along the course."),
        ],
    },
    "lifestyle/cooking_saffron_paella.pdf": {
        "title": "Cooking: Saffron Seafood Paella for Six",
        "subject": "cooking",
        "sections": [
            ("Ingredients",
             "For six servings you need 400 grams of bomba rice, 1.2 liters of fish "
             "stock, a generous pinch of saffron threads (about 0.5 grams), 12 large "
             "prawns, 500 grams of mussels, 300 grams of squid rings, one red bell "
             "pepper, two ripe grated tomatoes, four cloves of garlic, one teaspoon of "
             "smoked sweet paprika, 6 tablespoons of olive oil, salt, and two lemons."),
            ("Preparation",
             "Warm the fish stock and steep the saffron in it for at least 15 minutes. "
             "Clean the mussels and discard any that stay open when tapped. In a 38 "
             "centimeter paella pan, heat the olive oil and sear the prawns for one "
             "minute per side, then set them aside. Fry the squid and diced pepper for "
             "five minutes, add the garlic, then the grated tomato and paprika, and cook "
             "the sofrito until it darkens and thickens, about 8 minutes."),
            ("Cooking the Rice",
             "Scatter the rice evenly across the pan and stir it once to coat it in the "
             "sofrito. Pour in the hot saffron stock, season with salt, and from this "
             "moment do not stir again. Cook on high heat for 10 minutes, then lower the "
             "heat and nestle the mussels and prawns into the rice. Continue for another "
             "8 to 9 minutes until the liquid is absorbed."),
            ("The Socarrat",
             "The prized crispy layer at the bottom of the pan is called socarrat. To "
             "develop it, raise the heat for the final 90 seconds and listen for a gentle "
             "crackling sound. Smell carefully: toasted is good, burnt is not."),
            ("Resting and Serving",
             "Remove the pan from the heat, cover it with a clean kitchen towel, and let "
             "it rest for 5 minutes. Serve directly from the pan with lemon wedges. A "
             "chilled Albarino white wine is a classic pairing. Leftovers keep for one "
             "day in the refrigerator but the socarrat will lose its crunch."),
        ],
    },
    "lifestyle/travel_azores_itinerary.pdf": {
        "title": "Travel: Seven Days in the Azores",
        "subject": "travel",
        "sections": [
            ("Getting There",
             "The Azores are a Portuguese archipelago of nine volcanic islands in the "
             "middle of the Atlantic Ocean. Most visitors arrive at Ponta Delgada on Sao "
             "Miguel island; direct flights from Lisbon take about 2 hours 20 minutes. "
             "Inter-island flights are operated by the regional airline, and in summer "
             "ferries connect the central group of islands."),
            ("Days 1 to 3: Sao Miguel",
             "Spend the first day exploring Ponta Delgada and its black-and-white "
             "basalt pavements. On day two drive to Sete Cidades to see the twin lakes, "
             "one green and one blue, from the Vista do Rei viewpoint. On day three visit "
             "Furnas, where the local stew, cozido, is cooked for about six hours by "
             "burying the pot in volcanic ground, then bathe in the iron-rich thermal "
             "pools of Terra Nostra Garden."),
            ("Days 4 and 5: Pico",
             "Fly to Pico island, home to Mount Pico, which at 2,351 meters is the "
             "highest mountain in Portugal. The climb takes about 7 to 8 hours round trip "
             "and requires registering at the mountain house. On day five tour the "
             "lava-stone vineyards, a UNESCO World Heritage Site since 2004."),
            ("Days 6 and 7: Faial",
             "Take the 30-minute ferry to Faial. Walk around the Caldeira, a crater two "
             "kilometers wide, and visit the Capelinhos volcano, which erupted in 1957 "
             "and added new land to the island. Sailors crossing the Atlantic leave "
             "painted murals on the marina walls in Horta for good luck."),
            ("Practical Tips",
             "The best time to visit is from May to September, when average temperatures "
             "range from 20 to 25 degrees Celsius. Weather changes quickly, so pack a "
             "rain jacket. Whale watching is possible year-round, with the highest "
             "number of species seen in April and May. Renting a car is strongly "
             "recommended on Sao Miguel."),
        ],
    },
}


def build_pdf(path: Path, title: str, subject: str, sections) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=LETTER, title=title,
                            subject=subject, author="CSCI E-89 RAG exercise")
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    for i, (heading, text) in enumerate(sections):
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(Paragraph(text, styles["BodyText"]))
        story.append(Spacer(1, 10))
        if i == 2:  # make each file two pages long
            story.append(PageBreak())
    doc.build(story)


if __name__ == "__main__":
    for rel_path, spec in DOCS.items():
        out = DATA_DIR / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        build_pdf(out, spec["title"], spec["subject"], spec["sections"])
        print("wrote", out)
