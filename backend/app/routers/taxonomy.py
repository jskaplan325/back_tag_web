"""
Taxonomy Router - Areas of Law and Tag management.
"""
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel

from ..database.db import get_db, AreaOfLaw, Tag, TagUsage, Result

router = APIRouter()


# Pydantic schemas
class TagCreate(BaseModel):
    name: str
    description: Optional[str] = None
    patterns: List[str] = []
    semantic_examples: List[str] = []
    threshold: float = 0.45


class TagResponse(BaseModel):
    id: str
    area_of_law_id: str
    name: str
    description: Optional[str]
    patterns: List[str]
    semantic_examples: List[str]
    threshold: float
    created_at: datetime
    usage_count: int = 0
    avg_confidence: float = 0.0

    class Config:
        from_attributes = True


class AreaOfLawCreate(BaseModel):
    name: str
    description: Optional[str] = None
    color: str = "#3b82f6"
    icon: str = "Scale"


class AreaOfLawResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    color: str
    icon: str
    created_at: datetime
    tag_count: int = 0
    tags: List[TagResponse] = []

    class Config:
        from_attributes = True


class TagScoreboardEntry(BaseModel):
    tag_id: str
    tag_name: str
    area_of_law_name: str
    area_of_law_color: str
    usage_count: int
    avg_confidence: float
    document_percent: float  # % of documents with this tag


class ScoreboardResponse(BaseModel):
    total_documents: int
    total_tags: int
    total_areas: int
    top_tags: List[TagScoreboardEntry]


# Area of Law endpoints
@router.post("", response_model=AreaOfLawResponse)
async def create_area_of_law(
    aol: AreaOfLawCreate,
    db: Session = Depends(get_db)
):
    """Create a new Area of Law."""
    existing = db.query(AreaOfLaw).filter(AreaOfLaw.name == aol.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Area of Law already exists")

    aol_id = str(uuid.uuid4())
    db_aol = AreaOfLaw(
        id=aol_id,
        name=aol.name,
        description=aol.description,
        color=aol.color,
        icon=aol.icon
    )
    db.add(db_aol)
    db.commit()
    db.refresh(db_aol)

    return AreaOfLawResponse(
        id=db_aol.id,
        name=db_aol.name,
        description=db_aol.description,
        color=db_aol.color,
        icon=db_aol.icon,
        created_at=db_aol.created_at,
        tag_count=0,
        tags=[]
    )


@router.get("", response_model=List[AreaOfLawResponse])
async def list_areas_of_law(db: Session = Depends(get_db)):
    """List all Areas of Law with their tags."""
    areas = db.query(AreaOfLaw).order_by(AreaOfLaw.name).all()

    result = []
    for area in areas:
        tags_with_usage = []
        for tag in area.tags:
            usage_count = db.query(TagUsage).filter(TagUsage.tag_id == tag.id).count()
            avg_conf = db.query(func.avg(TagUsage.confidence)).filter(TagUsage.tag_id == tag.id).scalar() or 0
            tags_with_usage.append(TagResponse(
                id=tag.id,
                area_of_law_id=tag.area_of_law_id,
                name=tag.name,
                description=tag.description,
                patterns=tag.patterns or [],
                semantic_examples=tag.semantic_examples or [],
                threshold=tag.threshold,
                created_at=tag.created_at,
                usage_count=usage_count,
                avg_confidence=float(avg_conf)
            ))

        result.append(AreaOfLawResponse(
            id=area.id,
            name=area.name,
            description=area.description,
            color=area.color,
            icon=area.icon,
            created_at=area.created_at,
            tag_count=len(area.tags),
            tags=tags_with_usage
        ))

    return result


@router.get("/scoreboard", response_model=ScoreboardResponse)
async def get_scoreboard(db: Session = Depends(get_db)):
    """Get tag usage scoreboard."""
    # Get totals
    total_documents = db.query(Result).distinct(Result.document_id).count()
    total_tags = db.query(Tag).count()
    total_areas = db.query(AreaOfLaw).count()

    # Get tag usage stats
    tag_stats = db.query(
        Tag.id,
        Tag.name,
        AreaOfLaw.name.label('aol_name'),
        AreaOfLaw.color.label('aol_color'),
        func.count(TagUsage.id).label('usage_count'),
        func.avg(TagUsage.confidence).label('avg_confidence'),
        func.count(func.distinct(Result.document_id)).label('doc_count')
    ).outerjoin(
        TagUsage, Tag.id == TagUsage.tag_id
    ).outerjoin(
        Result, TagUsage.result_id == Result.id
    ).join(
        AreaOfLaw, Tag.area_of_law_id == AreaOfLaw.id
    ).group_by(
        Tag.id, Tag.name, AreaOfLaw.name, AreaOfLaw.color
    ).order_by(
        func.count(TagUsage.id).desc()
    ).limit(20).all()

    top_tags = []
    for stat in tag_stats:
        doc_percent = (stat.doc_count / total_documents * 100) if total_documents > 0 else 0
        top_tags.append(TagScoreboardEntry(
            tag_id=stat.id,
            tag_name=stat.name,
            area_of_law_name=stat.aol_name,
            area_of_law_color=stat.aol_color,
            usage_count=stat.usage_count or 0,
            avg_confidence=float(stat.avg_confidence or 0),
            document_percent=round(doc_percent, 1)
        ))

    return ScoreboardResponse(
        total_documents=total_documents,
        total_tags=total_tags,
        total_areas=total_areas,
        top_tags=top_tags
    )


@router.get("/{aol_id}", response_model=AreaOfLawResponse)
async def get_area_of_law(aol_id: str, db: Session = Depends(get_db)):
    """Get a specific Area of Law with its tags."""
    area = db.query(AreaOfLaw).filter(AreaOfLaw.id == aol_id).first()
    if not area:
        raise HTTPException(status_code=404, detail="Area of Law not found")

    tags_with_usage = []
    for tag in area.tags:
        usage_count = db.query(TagUsage).filter(TagUsage.tag_id == tag.id).count()
        avg_conf = db.query(func.avg(TagUsage.confidence)).filter(TagUsage.tag_id == tag.id).scalar() or 0
        tags_with_usage.append(TagResponse(
            id=tag.id,
            area_of_law_id=tag.area_of_law_id,
            name=tag.name,
            description=tag.description,
            patterns=tag.patterns or [],
            semantic_examples=tag.semantic_examples or [],
            threshold=tag.threshold,
            created_at=tag.created_at,
            usage_count=usage_count,
            avg_confidence=float(avg_conf)
        ))

    return AreaOfLawResponse(
        id=area.id,
        name=area.name,
        description=area.description,
        color=area.color,
        icon=area.icon,
        created_at=area.created_at,
        tag_count=len(area.tags),
        tags=tags_with_usage
    )


@router.patch("/{aol_id}", response_model=AreaOfLawResponse)
async def update_area_of_law(
    aol_id: str,
    update: AreaOfLawCreate,
    db: Session = Depends(get_db)
):
    """Update an Area of Law."""
    area = db.query(AreaOfLaw).filter(AreaOfLaw.id == aol_id).first()
    if not area:
        raise HTTPException(status_code=404, detail="Area of Law not found")

    area.name = update.name
    area.description = update.description
    area.color = update.color
    area.icon = update.icon
    db.commit()
    db.refresh(area)

    return await get_area_of_law(aol_id, db)


@router.delete("/{aol_id}")
async def delete_area_of_law(aol_id: str, db: Session = Depends(get_db)):
    """Delete an Area of Law and its tags."""
    area = db.query(AreaOfLaw).filter(AreaOfLaw.id == aol_id).first()
    if not area:
        raise HTTPException(status_code=404, detail="Area of Law not found")

    db.delete(area)
    db.commit()

    return {"message": "Area of Law deleted", "id": aol_id}


# Tag endpoints
@router.post("/{aol_id}/tags", response_model=TagResponse)
async def create_tag(
    aol_id: str,
    tag: TagCreate,
    db: Session = Depends(get_db)
):
    """Add a tag to an Area of Law."""
    area = db.query(AreaOfLaw).filter(AreaOfLaw.id == aol_id).first()
    if not area:
        raise HTTPException(status_code=404, detail="Area of Law not found")

    # Check for duplicate tag name within this area
    existing = db.query(Tag).filter(
        Tag.area_of_law_id == aol_id,
        Tag.name == tag.name
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Tag already exists in this Area of Law")

    tag_id = str(uuid.uuid4())
    db_tag = Tag(
        id=tag_id,
        area_of_law_id=aol_id,
        name=tag.name,
        description=tag.description,
        patterns=tag.patterns,
        semantic_examples=tag.semantic_examples,
        threshold=tag.threshold
    )
    db.add(db_tag)
    db.commit()
    db.refresh(db_tag)

    return TagResponse(
        id=db_tag.id,
        area_of_law_id=db_tag.area_of_law_id,
        name=db_tag.name,
        description=db_tag.description,
        patterns=db_tag.patterns or [],
        semantic_examples=db_tag.semantic_examples or [],
        threshold=db_tag.threshold,
        created_at=db_tag.created_at,
        usage_count=0,
        avg_confidence=0.0
    )


@router.get("/{aol_id}/tags/{tag_id}", response_model=TagResponse)
async def get_tag(aol_id: str, tag_id: str, db: Session = Depends(get_db)):
    """Get a specific tag."""
    tag = db.query(Tag).filter(Tag.id == tag_id, Tag.area_of_law_id == aol_id).first()
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")

    usage_count = db.query(TagUsage).filter(TagUsage.tag_id == tag.id).count()
    avg_conf = db.query(func.avg(TagUsage.confidence)).filter(TagUsage.tag_id == tag.id).scalar() or 0

    return TagResponse(
        id=tag.id,
        area_of_law_id=tag.area_of_law_id,
        name=tag.name,
        description=tag.description,
        patterns=tag.patterns or [],
        semantic_examples=tag.semantic_examples or [],
        threshold=tag.threshold,
        created_at=tag.created_at,
        usage_count=usage_count,
        avg_confidence=float(avg_conf)
    )


@router.patch("/{aol_id}/tags/{tag_id}", response_model=TagResponse)
async def update_tag(
    aol_id: str,
    tag_id: str,
    update: TagCreate,
    db: Session = Depends(get_db)
):
    """Update a tag."""
    tag = db.query(Tag).filter(Tag.id == tag_id, Tag.area_of_law_id == aol_id).first()
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")

    tag.name = update.name
    tag.description = update.description
    tag.patterns = update.patterns
    tag.semantic_examples = update.semantic_examples
    tag.threshold = update.threshold
    db.commit()
    db.refresh(tag)

    return await get_tag(aol_id, tag_id, db)


@router.delete("/{aol_id}/tags/{tag_id}")
async def delete_tag(aol_id: str, tag_id: str, db: Session = Depends(get_db)):
    """Delete a tag."""
    tag = db.query(Tag).filter(Tag.id == tag_id, Tag.area_of_law_id == aol_id).first()
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")

    db.delete(tag)
    db.commit()

    return {"message": "Tag deleted", "id": tag_id}


# Seed default taxonomy from back_tag
@router.post("/seed-defaults")
async def seed_default_taxonomy(db: Session = Depends(get_db)):
    """Seed the taxonomy with default Areas of Law and Tags from back_tag."""

    # Define default areas and their tags
    default_taxonomy = {
        "M&A / Corporate": {
            "description": "Mergers, acquisitions, and corporate transactions",
            "color": "#8b5cf6",  # purple
            "icon": "Building2",
            "tags": [
                {
                    "name": "Merger Agreement",
                    "description": "Merger and acquisition agreements",
                    "patterns": [
                        r'merger\s+agreement',
                        r'acquisition\s+agreement',
                        r'agreement\s+and\s+plan\s+of\s+merger',
                        r'business\s+combination',
                        r'stock\s+purchase\s+agreement',
                        r'asset\s+purchase\s+agreement',
                    ],
                    "semantic_examples": [
                        "agreement and plan of merger between companies",
                        "acquisition agreement for purchase of target company",
                        "definitive merger agreement terms and conditions",
                    ]
                },
                {
                    "name": "Due Diligence",
                    "description": "Transaction due diligence materials",
                    "patterns": [
                        r'due\s+diligence',
                        r'data\s+room',
                        r'diligence\s+materials?',
                        r'management\s+presentation',
                        r'information\s+memorandum',
                    ],
                    "semantic_examples": [
                        "due diligence materials and documentation",
                        "data room documents for transaction review",
                        "management presentation for potential buyers",
                    ]
                },
                {
                    "name": "Purchase Price",
                    "description": "Valuation and purchase price terms",
                    "patterns": [
                        r'purchase\s+price',
                        r'consideration',
                        r'valuation',
                        r'enterprise\s+value',
                        r'earnout',
                        r'escrow',
                    ],
                    "semantic_examples": [
                        "purchase price and consideration payable",
                        "enterprise value and equity valuation",
                        "earnout payments based on performance",
                    ]
                },
            ]
        },
        "Securities / Capital Markets": {
            "description": "Securities filings, offerings, and capital markets",
            "color": "#3b82f6",  # blue
            "icon": "TrendingUp",
            "tags": [
                {
                    "name": "Financial Statements",
                    "description": "Financial reports and statements",
                    "patterns": [
                        r'balance\s+sheet',
                        r'income\s+statement',
                        r'cash\s+flow\s+statement',
                        r'financial\s+statements?',
                        r'10-K',
                        r'10-Q',
                    ],
                    "semantic_examples": [
                        "financial statements showing assets and liabilities",
                        "balance sheet with total equity and debt",
                        "income statement with revenue and expenses",
                    ]
                },
                {
                    "name": "Risk Factors",
                    "description": "Risk disclosures and factors",
                    "patterns": [
                        r'risk\s+factors?',
                        r'risks?\s+and\s+uncertainties',
                        r'forward.looking\s+statements?',
                        r'material\s+risks?',
                    ],
                    "semantic_examples": [
                        "risk factors that could affect future performance",
                        "uncertainties and potential adverse effects",
                        "material risks to the business operations",
                    ]
                },
                {
                    "name": "Management Discussion",
                    "description": "MD&A and management analysis",
                    "patterns": [
                        r'management.s?\s+discussion',
                        r'MD&A',
                        r'management\s+analysis',
                        r'results\s+of\s+operations',
                    ],
                    "semantic_examples": [
                        "management's discussion and analysis of financial condition",
                        "MD&A section discussing results of operations",
                        "management analysis of liquidity and capital resources",
                    ]
                },
                {
                    "name": "Securities Filings",
                    "description": "SEC and regulatory filings",
                    "patterns": [
                        r'securities\s+and\s+exchange\s+commission',
                        r'SEC\s+filing',
                        r'form\s+S-1',
                        r'form\s+8-K',
                        r'proxy\s+statement',
                        r'prospectus',
                    ],
                    "semantic_examples": [
                        "SEC registration statement for securities offering",
                        "proxy statement for shareholder meeting",
                        "prospectus for initial public offering",
                    ]
                },
            ]
        },
        "Contracts / Commercial": {
            "description": "Commercial agreements and contract terms",
            "color": "#10b981",  # green
            "icon": "FileText",
            "tags": [
                {
                    "name": "Contract Terms",
                    "description": "General contract terms and provisions",
                    "patterns": [
                        r'terms?\s+and\s+conditions?',
                        r'representations?\s+and\s+warranties',
                        r'covenants?',
                        r'indemnification',
                        r'limitation\s+of\s+liability',
                        r'governing\s+law',
                    ],
                    "semantic_examples": [
                        "contractual terms and conditions of agreement",
                        "representations and warranties of the parties",
                        "indemnification obligations and procedures",
                    ]
                },
            ]
        },
    }

    created_areas = 0
    created_tags = 0

    for aol_name, aol_data in default_taxonomy.items():
        # Check if area exists
        existing_area = db.query(AreaOfLaw).filter(AreaOfLaw.name == aol_name).first()
        if existing_area:
            continue

        # Create area
        aol_id = str(uuid.uuid4())
        db_aol = AreaOfLaw(
            id=aol_id,
            name=aol_name,
            description=aol_data["description"],
            color=aol_data["color"],
            icon=aol_data["icon"]
        )
        db.add(db_aol)
        created_areas += 1

        # Create tags
        for tag_data in aol_data["tags"]:
            tag_id = str(uuid.uuid4())
            db_tag = Tag(
                id=tag_id,
                area_of_law_id=aol_id,
                name=tag_data["name"],
                description=tag_data["description"],
                patterns=tag_data["patterns"],
                semantic_examples=tag_data["semantic_examples"],
                threshold=0.45
            )
            db.add(db_tag)
            created_tags += 1

    db.commit()

    return {
        "message": "Default taxonomy seeded",
        "areas_created": created_areas,
        "tags_created": created_tags
    }
