# Environmental Dataset — column description

*(Plain-text copy of `environmental_data_description.docx`, from the course's Chapter 8 "Data Preprocessing Masterclass" zip. The **⚠ Audit note** lines were added after actually checking the data in `08-Data Preprocessing Step-by-Step Framework/scripting.py`. They're places where the original description and the real data disagree.)*

**File:** `environmental_data.csv`: 158 rows × 10 columns, daily readings from May to September 2010. It's the same data as R's classic `airquality` dataset (New York, 1973), relabelled to 2010, with deliberate data-quality problems added for practice.

| Column | Unit | Typical range / meaning |
|---|---|---|
| *(unnamed first column)* | – | Row number 1–158. An exported index, not a feature |
| **Ozone** | Parts per billion (ppb) | 0–50 = Good / safe · 51–100 = Moderate (sensitive people may feel irritation) · 101+ = Poor / unhealthy. The primary air-quality indicator: low = clean air, high = pollution and breathing discomfort |
| **Solar.R** | Watts per m² (W/m²) | 0–100 = very low sunlight (cloudy/evening) · 100–300 = moderate · 300–400+ = strong sunlight |
| **Wind** | Miles per hour (mph) | 0–5 = calm · 5–15 = light breeze · 15–25 = windy · 25+ = strong. Low wind means pollutants stay trapped; high wind disperses them |
| **Temp C** | Stated as °C | Stated ranges: <10 = cold · 10–25 = comfortable · 25–35 = hot · 35+ = extreme |
| **Temp** | – | "Duplicate column. Appears identical to Temp C" (redundant / earlier conversion / export mistake) |
| **Month** | 1–12 | Seasonal cycle |
| **Day** | 1–31 | Calendar day; used for daily trends / time-series sequencing |
| **Year** | Year number | Temporal grouping |
| **Weather** | Categorical | S = Sunny · PS = Partly Sunny · C = Cloudy |

**⚠ Audit notes (what the data actually shows):**
- **Temp C is *not* Celsius.** Its values run from 56 to 97, and 97 °C is impossible for outdoor air. These are **°F**, matching `airquality`'s Fahrenheit temperatures. `Temp C` equals `Temp` on all 157 numeric rows; the other row holds the typo **"C"** (row 11, where `Temp` = 74), which is why pandas reads the column as text.
- **Month** contains one text value, **"May"** (row 24), which is why it's read as text too.
- **Year** is **2010 on every row**, so it carries no information.
- **Duplicates:** rows 154–158 repeat the dates of rows 1, 149–152 (May 1; Sep 26–29) with identical measurements, but **4 of the 5 record a different Weather value**. Only one pair (rows 151 / 157) is an exact duplicate. There are **153 unique days**.
- **Missing values:** Ozone 38, Solar.R 7, Weather 3.
- **Outliers (1.5 × IQR rule):** Ozone 135 and 168 ppb, and Wind 20.1 and 20.7 mph. All are physically plausible readings, not errors.
