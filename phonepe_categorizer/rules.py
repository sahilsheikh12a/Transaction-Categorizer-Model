"""Stage 4 — deterministic keyword / brand rule engine.

A single ordered table. First match wins, so **specific patterns must come
before generic ones**. The ordering is load-bearing, not cosmetic:

    "GAJANAN MEDICAL GENERAL STORES"  -> HEALTH      (medical  before general store)
    "ISHANT DAIRY AND ICECREAM PARLOUR" -> FOOD      (icecream before dairy)
    "Shree Ganesh Book Depot ... General Sto" -> SHOPPING (book depot before general)
    "Roshan Mobile and Repairing Centre" -> SHOPPING (mobile repair before recharge)
    "A U K HOTELS PRIVATE LIMITED"    -> TRAVEL      (hotels ltd before bare hotel)

Two match kinds:
  PHRASE — substring match on the normalized merchant. Use for multi-word
           patterns and for brands whose name is a common word fragment.
  TOKEN  — whole-token match. Use for short words that would false-positive as
           a substring ("ola" inside "chocolate", "gas" inside "gases").

Curated from the real PhonePe statement in this repo. When adding entries,
add them next to their category block and keep specific-before-generic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from . import categories as C
from .normalize import canon, normalize_merchant

MatchKind = Literal["phrase", "token"]


@dataclass(frozen=True, slots=True)
class Rule:
    pattern: str
    category: str
    kind: MatchKind = "phrase"


def _p(pattern: str, category: str) -> Rule:
    return Rule(pattern, category, "phrase")


def _t(pattern: str, category: str) -> Rule:
    return Rule(pattern, category, "token")


# Phrases that are genuinely ambiguous in Indian statements. Matching one
# forces OTHER + needs_review instead of guessing. Checked before everything.
AMBIGUOUS_PHRASES: tuple[str, ...] = (
    "internet cafe",   # a computer/printing service, not food and not telecom
    "cyber cafe",
)


RULES: tuple[Rule, ...] = (
    # ══════════════════════════════════════════════════════════════════
    # EDUCATION, the colleges whose name contains a HEALTH word. Must
    # precede "medical" / "dental" / "nursing" / "ayurvedic" below. The rest
    # of EDUCATION is at the very end of the table.
    # ══════════════════════════════════════════════════════════════════
    _p("medical college", C.EDUCATION),
    _p("dental college", C.EDUCATION),
    _p("nursing college", C.EDUCATION),
    _p("ayurvedic college", C.EDUCATION),
    _p("pharmacy college", C.EDUCATION),
    _p("college of pharmacy", C.EDUCATION),

    # ══════════════════════════════════════════════════════════════════
    # HEALTH — before groceries, because "<name> MEDICAL AND GENERAL STORES"
    # is a pharmacy, not a kirana.
    # ══════════════════════════════════════════════════════════════════
    _p("medical store", C.HEALTH),
    _p("medical and general", C.HEALTH),
    _p("medical general", C.HEALTH),
    _t("medical", C.HEALTH),
    _t("medicals", C.HEALTH),
    _t("pharmacy", C.HEALTH),
    _t("pharma", C.HEALTH),
    _t("chemist", C.HEALTH),
    _t("chemists", C.HEALTH),
    _t("clinic", C.HEALTH),
    _t("hospital", C.HEALTH),
    _t("hospitals", C.HEALTH),
    _p("path labs", C.HEALTH),
    _p("patho", C.HEALTH),
    _t("diagnostics", C.HEALTH),
    _t("dental", C.HEALTH),
    _t("optical", C.HEALTH),
    _t("optician", C.HEALTH),
    _p("titan eye", C.HEALTH),          # eyewear chain — before bare "titan"
    _t("lenskart", C.HEALTH),
    _p("homoeopathic", C.HEALTH),
    _p("homeopathic", C.HEALTH),
    _p("ayurvedic", C.HEALTH),
    _p("mental health", C.HEALTH),
    _p("health institute", C.HEALTH),
    _p("healthcare", C.HEALTH),
    _p("nursing home", C.HEALTH),
    _t("apollo", C.HEALTH),
    _t("medplus", C.HEALTH),
    _p("1mg", C.HEALTH),
    _p("pharmeasy", C.HEALTH),
    _p("netmeds", C.HEALTH),

    # ══════════════════════════════════════════════════════════════════
    # MOBILE & INTERNET — before SHOPPING so "mobile repairing" can be
    # carved out first, and before BILLS so recharges land here.
    # ══════════════════════════════════════════════════════════════════
    _p("mobile repair", C.SHOPPING),
    _p("mobile and repairing", C.SHOPPING),
    _p("mobile recharge", C.MOBILE_AND_INTERNET),
    _p("prepaid recharge", C.MOBILE_AND_INTERNET),
    _p("postpaid", C.MOBILE_AND_INTERNET),
    _t("recharge", C.MOBILE_AND_INTERNET),
    _t("recharges", C.MOBILE_AND_INTERNET),
    _t("jio", C.MOBILE_AND_INTERNET),
    _t("airtel", C.MOBILE_AND_INTERNET),
    _p("vodafone", C.MOBILE_AND_INTERNET),
    _p("vodafoneidea", C.MOBILE_AND_INTERNET),
    _p("idea cellular", C.MOBILE_AND_INTERNET),
    _t("bsnl", C.MOBILE_AND_INTERNET),
    _t("broadband", C.MOBILE_AND_INTERNET),
    _p("act fibernet", C.MOBILE_AND_INTERNET),
    _p("hathway", C.MOBILE_AND_INTERNET),
    _p("excitel", C.MOBILE_AND_INTERNET),
    _p("net distribution services", C.MOBILE_AND_INTERNET),

    # ══════════════════════════════════════════════════════════════════
    # BILLS & UTILITIES
    # ══════════════════════════════════════════════════════════════════
    _t("electricity", C.BILLS_AND_UTILITIES),
    _p("state electricity", C.BILLS_AND_UTILITIES),
    _p("power distribution", C.BILLS_AND_UTILITIES),
    _t("mseb", C.BILLS_AND_UTILITIES),
    _t("msedcl", C.BILLS_AND_UTILITIES),
    _t("bescom", C.BILLS_AND_UTILITIES),
    _p("tata power", C.BILLS_AND_UTILITIES),
    _p("adani electricity", C.BILLS_AND_UTILITIES),
    _p("water bill", C.BILLS_AND_UTILITIES),
    _p("water works", C.BILLS_AND_UTILITIES),
    _p("municipal corporation", C.BILLS_AND_UTILITIES),
    _p("gas bill", C.BILLS_AND_UTILITIES),
    _p("gas limited", C.BILLS_AND_UTILITIES),
    _p("gas agency", C.BILLS_AND_UTILITIES),
    _p("indane", C.BILLS_AND_UTILITIES),
    _p("hp gas", C.BILLS_AND_UTILITIES),
    _p("bharatgas", C.BILLS_AND_UTILITIES),
    _p("dth recharge", C.BILLS_AND_UTILITIES),
    _p("tata sky", C.BILLS_AND_UTILITIES),
    _p("tatasky", C.BILLS_AND_UTILITIES),
    _p("dish tv", C.BILLS_AND_UTILITIES),
    _p("sun direct", C.BILLS_AND_UTILITIES),

    # ══════════════════════════════════════════════════════════════════
    # TRANSPORT — fuel, vehicle service, public transit
    # ══════════════════════════════════════════════════════════════════
    _p("petrol pump", C.TRANSPORT),
    _t("petrol", C.TRANSPORT),
    _t("petroleum", C.TRANSPORT),
    _t("diesel", C.TRANSPORT),
    _p("service station", C.TRANSPORT),
    _p("service stat", C.TRANSPORT),      # truncated in statements: "SERVICE STATIO"
    _p("filling station", C.TRANSPORT),
    _p("indian oil", C.TRANSPORT),
    _p("bharat petroleum", C.TRANSPORT),
    _p("hindustan petroleum", C.TRANSPORT),
    _t("hpcl", C.TRANSPORT),
    _t("bpcl", C.TRANSPORT),
    _t("iocl", C.TRANSPORT),
    _p("bp mobility", C.TRANSPORT),
    _p("rbml solutions", C.TRANSPORT),   # Reliance BP Mobility fuel retail
    _p("auto care", C.TRANSPORT),
    _p("auto service", C.TRANSPORT),
    _p("auto centre", C.TRANSPORT),
    _p("auto center", C.TRANSPORT),
    _p("auto parts", C.TRANSPORT),
    _p("auto mobile", C.TRANSPORT),
    _t("automobiles", C.TRANSPORT),
    _t("automobile", C.TRANSPORT),
    _t("tyres", C.TRANSPORT),
    _t("tyre", C.TRANSPORT),
    _p("lube service", C.TRANSPORT),
    _t("garage", C.TRANSPORT),
    _t("uber", C.TRANSPORT),
    _t("rapido", C.TRANSPORT),
    _p("ola cabs", C.TRANSPORT),
    _t("ola", C.TRANSPORT),
    _p("namma yatri", C.TRANSPORT),
    _t("irctc", C.TRANSPORT),
    _p("indian railway", C.TRANSPORT),
    _p("indian railways", C.TRANSPORT),
    _p("metro rail", C.TRANSPORT),
    _t("metro", C.TRANSPORT),
    _t("redbus", C.TRANSPORT),
    _t("fastag", C.TRANSPORT),
    _p("toll plaza", C.TRANSPORT),
    _t("parking", C.TRANSPORT),
    _p("state transport", C.TRANSPORT),
    _p("roadways", C.TRANSPORT),

    # ══════════════════════════════════════════════════════════════════
    # TRAVEL — corporate "HOTELS PVT LTD" before the bare "hotel" rule,
    # because in India a standalone "Hotel <name>" is nearly always an eatery.
    # ══════════════════════════════════════════════════════════════════
    _p("hotels private", C.TRAVEL),
    _p("hotels pvt", C.TRAVEL),
    _p("hotels limited", C.TRAVEL),
    _p("hotels ltd", C.TRAVEL),
    _p("resorts", C.TRAVEL),
    _p("resort", C.TRAVEL),
    _p("makemytrip", C.TRAVEL),
    _p("goibibo", C.TRAVEL),
    _p("cleartrip", C.TRAVEL),
    _p("yatra", C.TRAVEL),
    _p("easemytrip", C.TRAVEL),
    _p("oyo", C.TRAVEL),
    _p("air india", C.TRAVEL),
    _t("indigo", C.TRAVEL),
    _t("spicejet", C.TRAVEL),
    _t("vistara", C.TRAVEL),
    _t("airlines", C.TRAVEL),
    _t("airways", C.TRAVEL),
    _p("travels", C.TRAVEL),
    _p("tour and travel", C.TRAVEL),
    _p("tours and travels", C.TRAVEL),
    _p("holidays", C.TRAVEL),

    # ══════════════════════════════════════════════════════════════════
    # ENTERTAINMENT
    # ══════════════════════════════════════════════════════════════════
    _t("netflix", C.ENTERTAINMENT),
    _p("prime video", C.ENTERTAINMENT),
    _p("jiohotstar", C.ENTERTAINMENT),
    _t("hotstar", C.ENTERTAINMENT),
    _p("disney", C.ENTERTAINMENT),
    _p("sonyliv", C.ENTERTAINMENT),
    _t("zee5", C.ENTERTAINMENT),
    _t("spotify", C.ENTERTAINMENT),
    _p("youtube premium", C.ENTERTAINMENT),
    _p("apple music", C.ENTERTAINMENT),
    _p("bookmyshow", C.ENTERTAINMENT),
    _p("bigtree entertainment", C.ENTERTAINMENT),   # BookMyShow's legal entity
    _t("pvr", C.ENTERTAINMENT),
    _t("inox", C.ENTERTAINMENT),
    _t("cinema", C.ENTERTAINMENT),
    _t("cinemas", C.ENTERTAINMENT),
    _t("multiplex", C.ENTERTAINMENT),
    _p("film rent", C.ENTERTAINMENT),
    _p("films rent", C.ENTERTAINMENT),
    _p("rent house", C.ENTERTAINMENT),              # video/DVD rental shop
    _t("entertainment", C.ENTERTAINMENT),
    _p("sports and recreation", C.ENTERTAINMENT),
    _p("recreation", C.ENTERTAINMENT),
    _p("gaming", C.ENTERTAINMENT),

    # ══════════════════════════════════════════════════════════════════
    # FOOD & DINING — specific eatery types before the generic grocery
    # patterns, so "DAIRY AND ICECREAM PARLOUR" is dining, not groceries.
    # ══════════════════════════════════════════════════════════════════
    _t("swiggy", C.FOOD_AND_DINING),
    _t("zomato", C.FOOD_AND_DINING),
    _t("dunzo", C.FOOD_AND_DINING),
    _t("eatfit", C.FOOD_AND_DINING),
    _t("dominos", C.FOOD_AND_DINING),
    _p("pizza hut", C.FOOD_AND_DINING),
    _t("pizza", C.FOOD_AND_DINING),
    _p("mcdonald", C.FOOD_AND_DINING),
    _p("hardcastle restaurant", C.FOOD_AND_DINING),  # McDonald's West India
    _p("restaurant brands asia", C.FOOD_AND_DINING), # Burger King India
    _p("burger king", C.FOOD_AND_DINING),
    _t("burger", C.FOOD_AND_DINING),
    _t("kfc", C.FOOD_AND_DINING),
    _t("subway", C.FOOD_AND_DINING),
    _t("starbucks", C.FOOD_AND_DINING),
    _p("taco bell", C.FOOD_AND_DINING),
    _t("tacobell", C.FOOD_AND_DINING),
    _t("haldiram", C.FOOD_AND_DINING),
    _t("haldirams", C.FOOD_AND_DINING),
    _t("bikanerwala", C.FOOD_AND_DINING),
    _t("bikanervala", C.FOOD_AND_DINING),
    _p("cafe coffee day", C.FOOD_AND_DINING),
    _p("chai point", C.FOOD_AND_DINING),
    _p("ice cream", C.FOOD_AND_DINING),
    _t("icecream", C.FOOD_AND_DINING),
    _p("ice gola", C.FOOD_AND_DINING),
    _t("biryani", C.FOOD_AND_DINING),
    _t("dhaba", C.FOOD_AND_DINING),
    _t("restaurant", C.FOOD_AND_DINING),
    _t("restaurants", C.FOOD_AND_DINING),
    _t("resturant", C.FOOD_AND_DINING),   # sic — real spellings in the data
    _t("restorent", C.FOOD_AND_DINING),
    _t("cafe", C.FOOD_AND_DINING),
    _t("coffee", C.FOOD_AND_DINING),
    _t("bakery", C.FOOD_AND_DINING),
    _t("bakers", C.FOOD_AND_DINING),
    _t("canteen", C.FOOD_AND_DINING),
    _t("mess", C.FOOD_AND_DINING),
    _t("tiffin", C.FOOD_AND_DINING),
    _t("caterers", C.FOOD_AND_DINING),
    _t("catering", C.FOOD_AND_DINING),
    _t("sweets", C.FOOD_AND_DINING),
    _t("sweet", C.FOOD_AND_DINING),
    _t("misthan", C.FOOD_AND_DINING),
    _t("farsan", C.FOOD_AND_DINING),
    _p("far san", C.FOOD_AND_DINING),
    _t("namkeen", C.FOOD_AND_DINING),
    _p("chat center", C.FOOD_AND_DINING),
    _p("chat centre", C.FOOD_AND_DINING),
    _p("chaat", C.FOOD_AND_DINING),
    _p("chinese center", C.FOOD_AND_DINING),
    _p("chinese centre", C.FOOD_AND_DINING),
    _t("chinese", C.FOOD_AND_DINING),
    _t("chinescenter", C.FOOD_AND_DINING),
    _t("panipuri", C.FOOD_AND_DINING),
    _t("panipoori", C.FOOD_AND_DINING),
    _p("pani puri", C.FOOD_AND_DINING),
    _p("pav bhaji", C.FOOD_AND_DINING),
    _t("pohewala", C.FOOD_AND_DINING),
    _t("pohe", C.FOOD_AND_DINING),
    _t("nasta", C.FOOD_AND_DINING),
    _t("nashta", C.FOOD_AND_DINING),
    _p("tea stall", C.FOOD_AND_DINING),
    _p("tea stoll", C.FOOD_AND_DINING),
    _p("tea center", C.FOOD_AND_DINING),
    _p("tea centre", C.FOOD_AND_DINING),
    _t("tea", C.FOOD_AND_DINING),
    _p("juice center", C.FOOD_AND_DINING),
    _p("juice centre", C.FOOD_AND_DINING),
    _t("juice", C.FOOD_AND_DINING),
    _t("sharbat", C.FOOD_AND_DINING),
    _p("coconut water", C.FOOD_AND_DINING),
    _p("pan palace", C.FOOD_AND_DINING),
    _p("pan shop", C.FOOD_AND_DINING),
    _p("paan", C.FOOD_AND_DINING),
    _p("eating point", C.FOOD_AND_DINING),
    _p("food court", C.FOOD_AND_DINING),
    # Bare "food"/"foods" as a whole token — "Aditya Food", "Sharma Foods".
    # Sits inside the dining block so a grocery pattern later cannot claim it.
    _t("food", C.FOOD_AND_DINING),
    _t("foods", C.FOOD_AND_DINING),
    _p("nepenthe coffee", C.FOOD_AND_DINING),
    _p("chocolates", C.FOOD_AND_DINING),
    _t("hotel", C.FOOD_AND_DINING),      # after the "hotels pvt ltd" rules
    _t("hotels", C.FOOD_AND_DINING),

    # ── books & stationery, hoisted above GROCERIES ──────────────────
    # "Shree Ganesh Book Depot ... and General Sto" is a bookshop that also
    # sells sundries; the head noun wins, so these must beat _t("general").
    _p("book depot", C.SHOPPING),
    _p("book house", C.SHOPPING),
    _p("book stall", C.SHOPPING),
    _t("books", C.SHOPPING),
    _t("stationery", C.SHOPPING),
    _t("stationary", C.SHOPPING),

    # ══════════════════════════════════════════════════════════════════
    # GROCERIES — kirana / daily needs / staples / fresh produce
    # ══════════════════════════════════════════════════════════════════
    _t("kirana", C.GROCERIES),
    _p("daily needs", C.GROCERIES),
    _p("daily and general", C.GROCERIES),
    _p("general store", C.GROCERIES),
    _p("general stores", C.GROCERIES),
    _t("general", C.GROCERIES),
    _t("bhandar", C.GROCERIES),
    _t("provision", C.GROCERIES),
    _t("provisions", C.GROCERIES),
    _t("supermarket", C.GROCERIES),
    _t("supermart", C.GROCERIES),
    _t("supermarts", C.GROCERIES),        # AVENUE SUPERMARTS = DMart
    _t("dmart", C.GROCERIES),
    _p("mega mart", C.GROCERIES),
    _t("megamart", C.GROCERIES),
    _p("reliance fresh", C.GROCERIES),
    _p("reliance retail", C.GROCERIES),
    _p("big bazaar", C.GROCERIES),
    _p("big bazar", C.GROCERIES),
    _t("bazar", C.GROCERIES),
    _t("bazaar", C.GROCERIES),
    _p("more megastore", C.GROCERIES),
    _t("bigbasket", C.GROCERIES),
    _t("blinkit", C.GROCERIES),
    _t("zepto", C.GROCERIES),
    _t("instamart", C.GROCERIES),
    _p("jiomart", C.GROCERIES),
    _p("amazon pay groceries", C.GROCERIES),
    _t("groceries", C.GROCERIES),
    _t("grocery", C.GROCERIES),
    _t("dairy", C.GROCERIES),
    _t("milk", C.GROCERIES),
    _p("fruit center", C.GROCERIES),
    _p("fruit centre", C.GROCERIES),
    _p("fruits center", C.GROCERIES),
    _p("fruit shop", C.GROCERIES),
    _t("fruits", C.GROCERIES),
    _t("fruit", C.GROCERIES),
    _t("vegetable", C.GROCERIES),
    _t("vegetables", C.GROCERIES),
    _t("sabzi", C.GROCERIES),
    _p("chicken and mutton", C.GROCERIES),
    _t("chicken", C.GROCERIES),
    _t("mutton", C.GROCERIES),
    _p("fish center", C.GROCERIES),
    _p("fish centre", C.GROCERIES),
    _p("egg point", C.GROCERIES),
    _p("flour and besan", C.GROCERIES),
    _p("flour mill", C.GROCERIES),
    _t("besan", C.GROCERIES),

    # ══════════════════════════════════════════════════════════════════
    # SHOPPING — marketplaces, apparel, electronics, books/stationery
    # ══════════════════════════════════════════════════════════════════
    _p("amazon seller", C.SHOPPING),
    _p("amazon pay", C.SHOPPING),
    _t("amazon", C.SHOPPING),
    _t("flipkart", C.SHOPPING),
    _t("myntra", C.SHOPPING),
    _t("ajio", C.SHOPPING),
    _t("nykaa", C.SHOPPING),
    _t("meesho", C.SHOPPING),
    _t("snapdeal", C.SHOPPING),
    _p("shoppers stop", C.SHOPPING),
    _t("westside", C.SHOPPING),
    _p("max fashion", C.SHOPPING),
    _t("croma", C.SHOPPING),
    _p("reliance digital", C.SHOPPING),
    _p("vijay sales", C.SHOPPING),
    # Brand stores. National brands a statement shows as "<brand> store",
    # "<brand> smart plaza" or "<brand> exclusive". Token rules only where the
    # word cannot be a person's name or another business: "zara", "raymond",
    # "philips" and "biba" are common names, "hp" is also a fuel brand, and a
    # bare "apple" is a fruit stall — those get phrase rules or none.
    _t("samsung", C.SHOPPING),
    _t("realme", C.SHOPPING),
    _t("xiaomi", C.SHOPPING),
    _t("redmi", C.SHOPPING),
    _t("poco", C.SHOPPING),
    _p("mi store", C.SHOPPING),
    _p("mi home", C.SHOPPING),
    _t("oppo", C.SHOPPING),
    _t("vivo", C.SHOPPING),
    _t("iqoo", C.SHOPPING),
    _t("oneplus", C.SHOPPING),
    _p("one plus", C.SHOPPING),
    _t("motorola", C.SHOPPING),
    _t("nokia", C.SHOPPING),
    _p("apple store", C.SHOPPING),
    _p("apple india", C.SHOPPING),
    _t("lenovo", C.SHOPPING),
    _t("asus", C.SHOPPING),
    _t("acer", C.SHOPPING),
    _t("panasonic", C.SHOPPING),
    _t("whirlpool", C.SHOPPING),
    _t("haier", C.SHOPPING),
    _t("voltas", C.SHOPPING),
    _t("havells", C.SHOPPING),
    _p("philips india", C.SHOPPING),
    _p("boat lifestyle", C.SHOPPING),
    _p("sangeetha mobiles", C.SHOPPING),
    _t("poorvika", C.SHOPPING),
    _t("nike", C.SHOPPING),
    _t("adidas", C.SHOPPING),
    _t("puma", C.SHOPPING),
    _t("reebok", C.SHOPPING),
    _t("skechers", C.SHOPPING),
    _t("bata", C.SHOPPING),
    _p("red tape", C.SHOPPING),
    _t("woodland", C.SHOPPING),
    _p("metro shoes", C.SHOPPING),
    _t("levis", C.SHOPPING),
    _t("zudio", C.SHOPPING),
    _t("pantaloons", C.SHOPPING),
    _t("uniqlo", C.SHOPPING),
    _p("allen solly", C.SHOPPING),
    _p("peter england", C.SHOPPING),
    _p("van heusen", C.SHOPPING),
    _p("louis philippe", C.SHOPPING),
    _t("manyavar", C.SHOPPING),
    _t("fabindia", C.SHOPPING),
    _t("jockey", C.SHOPPING),
    _t("wrogn", C.SHOPPING),
    _t("bewakoof", C.SHOPPING),
    _t("titan", C.SHOPPING),
    _t("fastrack", C.SHOPPING),
    _t("tanishq", C.SHOPPING),
    _t("caratlane", C.SHOPPING),
    _p("kalyan jewellers", C.SHOPPING),
    _p("malabar gold", C.SHOPPING),
    _t("ikea", C.SHOPPING),
    _t("decathlon", C.SHOPPING),
    _t("xerox", C.SHOPPING),
    _p("photocopy", C.SHOPPING),
    _t("garment", C.SHOPPING),
    _t("garments", C.SHOPPING),
    _t("saree", C.SHOPPING),
    _t("sadi", C.SHOPPING),
    _t("textiles", C.SHOPPING),
    _t("footwear", C.SHOPPING),
    _p("boot house", C.SHOPPING),
    _t("boots", C.SHOPPING),
    _t("trends", C.SHOPPING),
    _t("boutique", C.SHOPPING),
    _t("fashion", C.SHOPPING),
    _t("collection", C.SHOPPING),
    _p("souled store", C.SHOPPING),
    _p("streetstyle", C.SHOPPING),
    _t("electronics", C.SHOPPING),
    _t("electronic", C.SHOPPING),
    _t("electrical", C.SHOPPING),
    _p("electric stores", C.SHOPPING),
    _t("furniture", C.SHOPPING),
    _p("cycle stores", C.SHOPPING),
    _t("hardware", C.SHOPPING),
    _p("sony center", C.SHOPPING),
    _p("sony centre", C.SHOPPING),
    _p("digital home", C.SHOPPING),
    _t("novelty", C.SHOPPING),
    _p("gift shop", C.SHOPPING),
    _t("toys", C.SHOPPING),
    _p("dry cleaning", C.SHOPPING),
    _t("laundry", C.SHOPPING),
    _t("salon", C.SHOPPING),
    _t("parlour", C.SHOPPING),
    _p("beauty parlor", C.SHOPPING),

    # ══════════════════════════════════════════════════════════════════
    # FINANCE
    # ══════════════════════════════════════════════════════════════════
    _p("insurance", C.FINANCE),
    _p("life insurance", C.FINANCE),
    _t("lic", C.FINANCE),
    _p("mutual fund", C.FINANCE),
    _t("groww", C.FINANCE),
    _t("zerodha", C.FINANCE),
    _t("upstox", C.FINANCE),
    _p("bank charges", C.FINANCE),
    _p("service charge", C.FINANCE),
    _p("loan repayment", C.FINANCE),
    _p("emi payment", C.FINANCE),
    _p("credit card payment", C.FINANCE),
    _p("bajaj finance", C.FINANCE),
    _p("bajaj finserv", C.FINANCE),

    # ══════════════════════════════════════════════════════════════════
    # EDUCATION — deliberately LAST. Merchant names carry addresses
    # ("Sharma Kirana College Road", "Juice Centre School Square"), so every
    # other keyword must get the first chance; these fire only when nothing
    # else in the string says what the business is. "health institute" is
    # HEALTH for the same reason: the HEALTH block has already claimed it.
    # ══════════════════════════════════════════════════════════════════
    _p("exam fee", C.EDUCATION),
    _p("examination fee", C.EDUCATION),
    _p("admission fee", C.EDUCATION),
    _p("tuition fee", C.EDUCATION),
    _p("physics wallah", C.EDUCATION),
    _t("physicswallah", C.EDUCATION),
    _t("byjus", C.EDUCATION),
    _t("byju", C.EDUCATION),
    _t("unacademy", C.EDUCATION),
    _t("vedantu", C.EDUCATION),
    _t("udemy", C.EDUCATION),
    _t("coursera", C.EDUCATION),
    _t("upgrad", C.EDUCATION),
    _t("simplilearn", C.EDUCATION),
    _t("fiitjee", C.EDUCATION),
    _p("allen career", C.EDUCATION),
    _t("university", C.EDUCATION),
    _t("college", C.EDUCATION),
    _t("colleges", C.EDUCATION),
    _t("school", C.EDUCATION),
    _t("schools", C.EDUCATION),
    _t("vidyalaya", C.EDUCATION),
    _t("vidyalay", C.EDUCATION),
    _t("mahavidyalaya", C.EDUCATION),
    _t("vidyapeeth", C.EDUCATION),
    _t("convent", C.EDUCATION),
    _t("polytechnic", C.EDUCATION),
    _t("academy", C.EDUCATION),
    _t("institute", C.EDUCATION),
    _t("coaching", C.EDUCATION),
    _t("tuition", C.EDUCATION),
    _t("tuitions", C.EDUCATION),
    _t("tutorials", C.EDUCATION),
    _t("classes", C.EDUCATION),
)


# Confidence per source layer. Rules are curated by hand, so they sit just
# below an explicit user override and an exact merchant-directory hit.
RULE_CONFIDENCE = 0.90


def lookup(signal: str | None) -> tuple[str, float, str] | None:
    """Return (category, confidence, matched_pattern), or None if no rule fires.

    The signal is normalized with `normalize_merchant` first, so callers may
    pass the raw statement string.
    """
    if not signal:
        return None
    needle = normalize_merchant(signal)
    if not needle:
        return None

    # `normalize_merchant` drops corporate-form tokens, which is right for
    # fuzzy matching but hides real signal from phrase rules: "A U K HOTELS
    # PRIVATE LIMITED" must reach the "hotels private" -> TRAVEL rule before
    # the bare "hotel" -> FOOD rule. So phrase rules get a second, unstripped
    # haystack. Token rules keep using the stripped token set, where the
    # corporate words would only add noise.
    needle_full = canon(signal)

    if _ambiguous(needle, needle_full):
        return None  # force fallthrough to OTHER + needs_review

    tokens = set(needle.split())
    for rule in RULES:
        if rule.kind == "token":
            if rule.pattern in tokens:
                return rule.category, RULE_CONFIDENCE, rule.pattern
        elif rule.pattern in needle or rule.pattern in needle_full:
            return rule.category, RULE_CONFIDENCE, rule.pattern
    return None


def _ambiguous(needle: str, needle_full: str) -> bool:
    return any(p in needle or p in needle_full for p in AMBIGUOUS_PHRASES)


def is_ambiguous(signal: str | None) -> bool:
    """Does the string contain a phrase the rules deliberately abstain on?

    Later layers use this to abstain too, rather than guess where the rule
    table already decided a guess is wrong.
    """
    if not signal:
        return False
    return _ambiguous(normalize_merchant(signal), canon(signal))


def rule_count() -> int:
    return len(RULES)


__all__ = ["RULES", "Rule", "lookup", "is_ambiguous", "rule_count", "RULE_CONFIDENCE", "AMBIGUOUS_PHRASES"]
