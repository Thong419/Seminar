"""Curated demo retrieval provider for stable seminar presentations."""

from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.document_fetcher import EvidenceDocument
from src.retrieval.search_client import SearchResult
from src.retrieval.source_credibility import score_source_credibility


@dataclass(frozen=True)
class DemoCorpusEntry:
    article_key: str
    title: str
    url: str
    source: str
    snippet: str
    content: str
    provider_relevance: float


DEMO_CORPUS: list[DemoCorpusEntry] = [
    DemoCorpusEntry(
        article_key="trump_bannon",
        title="Trump removes Bannon from NSC role in shake-up",
        url="https://www.reuters.com/world/us/trump-bannon-nsc",
        source="Reuters",
        snippet="Reuters reported that Trump removed Steve Bannon from the National Security Council.",
        content=(
            "Reuters reported that the White House confirmed President Donald Trump removed Steve Bannon "
            "from the National Security Council during a routine restructuring. The White House said the "
            "change improved coordination of foreign policy and national security decisions."
        ),
        provider_relevance=0.98,
    ),
    DemoCorpusEntry(
        article_key="trump_bannon",
        title="White House restructuring statement on National Security Council",
        url="https://www.whitehouse.gov/briefing-room/statement/nsc-restructuring",
        source="whitehouse.gov",
        snippet="White House statement on NSC restructuring.",
        content=(
            "The White House announced a routine restructuring of the National Security Council. The "
            "statement confirms Steve Bannon was removed from the principals committee to improve policy coordination."
        ),
        provider_relevance=0.95,
    ),
    DemoCorpusEntry(
        article_key="trump_bannon",
        title="White House",
        url="https://en.wikipedia.org/wiki/White_House",
        source="Wikipedia",
        snippet="The White House is the official residence of the president.",
        content="The White House is the official residence and workplace of the president of the United States in Washington, D.C.",
        provider_relevance=0.62,
    ),
    DemoCorpusEntry(
        article_key="trump_bannon",
        title="Donald Trump",
        url="https://en.wikipedia.org/wiki/Donald_Trump",
        source="Wikipedia",
        snippet="Donald Trump is an American politician and businessman.",
        content="Donald Trump is an American politician, media personality, and businessman who served as the 45th president of the United States.",
        provider_relevance=0.60,
    ),
    DemoCorpusEntry(
        article_key="trump_bannon",
        title="U.S. domestic politics overview",
        url="https://example.com/politics-overview",
        source="example.com",
        snippet="Political reactions across Washington.",
        content="A broad overview of political reactions in Washington without discussing Steve Bannon or the National Security Council change.",
        provider_relevance=0.48,
    ),
    DemoCorpusEntry(
        article_key="perseverance",
        title="NASA confirms Perseverance collected Mars rock sample at Jezero Crater",
        url="https://www.nasa.gov/missions/mars2020/perseverance-sample-jezero",
        source="nasa.gov",
        snippet="NASA confirmed Perseverance collected a rock core sample from Jezero Crater.",
        content=(
            "NASA confirmed that the Perseverance rover successfully collected a Mars rock core sample "
            "from Jezero Crater. The agency verified the sample collection after reviewing telemetry and imaging data from Mars."
        ),
        provider_relevance=0.99,
    ),
    DemoCorpusEntry(
        article_key="perseverance",
        title="Science mission update on Mars sample collection",
        url="https://www.science.org/content/article/mars-sample-collection-perseverance",
        source="science.org",
        snippet="Science magazine reports successful sample collection by Perseverance.",
        content=(
            "Science reported that NASA's Perseverance rover collected rock samples from Jezero Crater on Mars. "
            "Researchers verified the sample tube contents during the mission update."
        ),
        provider_relevance=0.94,
    ),
    DemoCorpusEntry(
        article_key="perseverance",
        title="Mars",
        url="https://en.wikipedia.org/wiki/Mars",
        source="Wikipedia",
        snippet="Mars is the fourth planet from the Sun.",
        content="Mars is the fourth planet from the Sun and has two moons. It is often called the Red Planet.",
        provider_relevance=0.58,
    ),
    DemoCorpusEntry(
        article_key="perseverance",
        title="Jezero Crater",
        url="https://en.wikipedia.org/wiki/Jezero_Crater",
        source="Wikipedia",
        snippet="Jezero is a crater on Mars.",
        content="Jezero Crater is an impact crater on Mars believed to have once hosted a lake.",
        provider_relevance=0.57,
    ),
    DemoCorpusEntry(
        article_key="cancer_hoax",
        title="No evidence that warm salt water cures cancer",
        url="https://www.cancer.gov/news-events/cancer-currents-blog/salt-water-cancer-misinformation",
        source="cancer.gov",
        snippet="Medical experts say there is no evidence that salt water cures cancer.",
        content=(
            "Cancer research organizations and medical treatment experts state there is no evidence that drinking warm "
            "salt water cures cancer within 3 days. The claim is misinformation and no cancer treatment guidelines support it."
        ),
        provider_relevance=0.99,
    ),
    DemoCorpusEntry(
        article_key="cancer_hoax",
        title="WHO warns against false cancer cure misinformation",
        url="https://www.who.int/news-room/fact-check/cancer-cure-misinformation",
        source="who.int",
        snippet="WHO warns that false cure claims can harm patients.",
        content=(
            "The World Health Organization warns that false claims about medical treatment and miracle cancer cures are misinformation. "
            "There is no scientific evidence that warm salt water can cure cancer without medical treatment."
        ),
        provider_relevance=0.95,
    ),
    DemoCorpusEntry(
        article_key="cancer_hoax",
        title="Great Salt Lake",
        url="https://en.wikipedia.org/wiki/Great_Salt_Lake",
        source="Wikipedia",
        snippet="The Great Salt Lake is a saline lake in Utah.",
        content="The Great Salt Lake is a saline lake in Utah known for geography, climate, and wildlife.",
        provider_relevance=0.61,
    ),
    DemoCorpusEntry(
        article_key="cancer_hoax",
        title="Climate change and salt lakes",
        url="https://example.com/climate-salt-lakes",
        source="example.com",
        snippet="Climate change affects lake salinity.",
        content="A geography article about Utah lake salinity and climate change trends without any cancer research or medical treatment discussion.",
        provider_relevance=0.47,
    ),
    DemoCorpusEntry(
        article_key="smartphone_hoax",
        title="Fact check: no worldwide government smartphone ban announced",
        url="https://www.factcheck.org/2026/06/no-worldwide-smartphone-ban",
        source="factcheck.org",
        snippet="Fact-checkers found no official policy banning smartphones worldwide.",
        content=(
            "Fact-checkers found no official government policy announcing a worldwide smartphone ban next month. "
            "The claim is false and no policy verification source supports a secret ban to reduce dependency on technology."
        ),
        provider_relevance=0.98,
    ),
    DemoCorpusEntry(
        article_key="smartphone_hoax",
        title="Government policy offices deny smartphone ban rumor",
        url="https://www.reuters.com/world/no-smartphone-ban-policy-rumor",
        source="Reuters",
        snippet="Officials deny any secret global smartphone ban announcement.",
        content=(
            "Reuters reported that government policy officials denied any secret announcement banning smartphones worldwide. "
            "No official verification source confirms the claim, and the rumor was described as misinformation."
        ),
        provider_relevance=0.96,
    ),
    DemoCorpusEntry(
        article_key="smartphone_hoax",
        title="Smartphone",
        url="https://en.wikipedia.org/wiki/Smartphone",
        source="Wikipedia",
        snippet="A smartphone is a portable computer device.",
        content="A smartphone is a portable computing device combining mobile telephone and computer functions.",
        provider_relevance=0.63,
    ),
    DemoCorpusEntry(
        article_key="smartphone_hoax",
        title="Huawei",
        url="https://en.wikipedia.org/wiki/Huawei",
        source="Wikipedia",
        snippet="Huawei is a technology company.",
        content="Huawei is a Chinese multinational technology corporation that designs telecommunications equipment and consumer electronics.",
        provider_relevance=0.60,
    ),
    DemoCorpusEntry(
        article_key="smartphone_hoax",
        title="Nokia",
        url="https://en.wikipedia.org/wiki/Nokia",
        source="Wikipedia",
        snippet="Nokia is a Finnish company.",
        content="Nokia is a Finnish multinational telecommunications, information technology, and consumer electronics company.",
        provider_relevance=0.60,
    ),
    DemoCorpusEntry(
        article_key="smartphone_hoax",
        title="WhatsApp",
        url="https://en.wikipedia.org/wiki/WhatsApp",
        source="Wikipedia",
        snippet="WhatsApp is an instant messaging service.",
        content="WhatsApp is an instant messaging and voice-over-IP service owned by Meta.",
        provider_relevance=0.59,
    ),
]


class DemoSearchClient:
    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        article_key = self._match_article(query)
        if article_key is None:
            return []
        rows = [row for row in DEMO_CORPUS if row.article_key == article_key]
        lowered = query.lower()
        if article_key == "smartphone_hoax":
            if "fact check" in lowered or "fact-check" in lowered:
                preferred = [
                    "Fact check: no worldwide government smartphone ban announced",
                    "Government policy offices deny smartphone ban rumor",
                    "Smartphone",
                    "Huawei",
                    "WhatsApp",
                ]
            else:
                preferred = [
                    "Fact check: no worldwide government smartphone ban announced",
                    "Government policy offices deny smartphone ban rumor",
                    "Smartphone",
                    "Nokia",
                    "WhatsApp",
                ]
            rows = [row for title in preferred for row in rows if row.title == title]
        return [
            SearchResult(
                title=row.title,
                url=row.url,
                source=row.source,
                snippet=row.snippet,
                provider_relevance=row.provider_relevance,
                query=query,
                provider="demo_search",
            )
            for row in rows[:limit]
        ]

    def _match_article(self, query: str) -> str | None:
        lowered = query.lower()
        if any(token in lowered for token in ("bannon", "security council", "nsc", "trump")):
            return "trump_bannon"
        if any(token in lowered for token in ("perseverance", "jezero", "mars sample", "rock samples")):
            return "perseverance"
        if any(token in lowered for token in ("salt water", "cancer", "medical treatment")):
            return "cancer_hoax"
        if any(token in lowered for token in ("smartphone", "banned", "government")):
            return "smartphone_hoax"
        return None


class DemoDocumentFetcher:
    def fetch(self, result: SearchResult) -> EvidenceDocument:
        row = next(row for row in DEMO_CORPUS if row.url == result.url)
        return EvidenceDocument(
            title=row.title,
            url=row.url,
            source=row.source,
            content=row.content,
            trust_score=0.0,
            relevance_score=result.provider_relevance,
            query=result.query,
            provider=result.provider,
            source_credibility=score_source_credibility(row.source, row.url),
        )


def build_demo_retrieval_components() -> tuple[DemoSearchClient, DemoDocumentFetcher]:
    return DemoSearchClient(), DemoDocumentFetcher()
