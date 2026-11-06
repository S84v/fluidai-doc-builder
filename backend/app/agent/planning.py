"""Request understanding and document planning."""

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.agent.models import DocumentPlan, DocumentType
from backend.app.services.llm import build_chat_model

DOCUMENT_TYPE_GUIDANCE: dict[DocumentType, str] = {
    DocumentType.BUSINESS_PROPOSAL: (
        "proposes a solution or course of action and seeks approval"
    ),
    DocumentType.MEETING_MINUTES: (
        "records meeting details, discussion, decisions and action items"
    ),
    DocumentType.PROJECT_PLAN: (
        "describes project execution, including tasks, milestones, "
        "responsibilities and timelines"
    ),
    DocumentType.BUSINESS_REPORT: (
        "communicates findings, analysis, results and recommendations"
    ),
    DocumentType.TECHNICAL_DESIGN: (
        "describes the architecture, components and technical implementation "
        "of a solution"
    ),
    DocumentType.SOP: "documents repeatable procedures and operational instructions",
    DocumentType.PRODUCT_SPECIFICATION: (
        "defines product requirements, behaviour, users and acceptance criteria"
    ),
    DocumentType.OTHER: (
        "used when no existing category is suitable or the document type cannot "
        "be determined reliably"
    ),
}

DOCUMENT_STRUCTURE_GUIDANCE: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.BUSINESS_PROPOSAL: (
        "Executive Summary",
        "Problem or Opportunity",
        "Proposed Solution",
        "Scope and Deliverables",
        "Implementation Approach",
        "Timeline",
        "Investment and Costs",
        "Benefits and Risks",
        "Next Steps",
    ),
    DocumentType.MEETING_MINUTES: (
        "Meeting Details",
        "Agenda",
        "Discussion Points",
        "Decisions",
        "Action Items",
    ),
    DocumentType.PROJECT_PLAN: (
        "Project Overview",
        "Objectives",
        "Scope and Deliverables",
        "Workstreams",
        "Milestones and Timeline",
        "Responsibilities",
        "Dependencies",
        "Risks and Mitigation",
        "Success Criteria",
    ),
    DocumentType.BUSINESS_REPORT: (
        "Executive Summary",
        "Background and Scope",
        "Methodology",
        "Findings",
        "Analysis",
        "Recommendations",
        "Conclusion",
        "Appendices",
    ),
    DocumentType.TECHNICAL_DESIGN: (
        "Context and Objectives",
        "Requirements",
        "Architecture Overview",
        "Components and Interfaces",
        "Data Flows",
        "Security and Reliability",
        "Failure Handling",
        "Deployment and Testing",
        "Technical Decisions and Open Questions",
    ),
    DocumentType.SOP: (
        "Purpose and Scope",
        "Roles and Responsibilities",
        "Prerequisites",
        "Safety and Operational Controls",
        "Procedure Steps",
        "Exceptions and Troubleshooting",
        "Escalation",
        "Records and References",
    ),
    DocumentType.PRODUCT_SPECIFICATION: (
        "Product Overview",
        "Problem Statement",
        "Target Users",
        "Goals and Non-Goals",
        "Functional Requirements",
        "Non-Functional Requirements",
        "User Workflows",
        "Acceptance Criteria",
        "Success Metrics",
        "Dependencies and Open Questions",
    ),
    DocumentType.OTHER: (
        "Executive Overview",
        "Context and Background",
        "Main Content",
        "Considerations",
        "Recommendations and Actions",
        "Conclusion and Next Steps",
    ),
}

_CLASSIFICATION_GUIDANCE = "\n".join(
    f"- {document_type.value}: {guidance}"
    for document_type, guidance in DOCUMENT_TYPE_GUIDANCE.items()
)
_STRUCTURE_GUIDANCE = "\n".join(
    f"- {document_type.value}: {', '.join(sections)}"
    for document_type, sections in DOCUMENT_STRUCTURE_GUIDANCE.items()
)
_JSON_SCHEMA = json.dumps(DocumentPlan.model_json_schema(), indent=2)

PLANNING_SYSTEM_PROMPT = f"""You are a document-planning assistant. You plan
structured business documents; you do not write their content.

Classify the request into exactly one document_type.

Classification guidance:
{_CLASSIFICATION_GUIDANCE}

Classify using the requested document's primary purpose, intended audience,
desired outcome, nature of the information being communicated, and expected
structure of the final document. Do not classify using isolated keywords alone.
When multiple types are plausible, choose the type that best represents the
primary purpose of the requested deliverable.
Use other when no existing category is suitable or the type cannot be
determined reliably; do not make an arbitrary classification.

Suggested section structures by document type:
{_STRUCTURE_GUIDANCE}

Use these structures as guidance for the selected type. Adapt them to the
request and omit sections that are irrelevant or unsupported by known facts.

Use the following JSON schema for your response:
{_JSON_SCHEMA}

Planning rules:
- Use only facts explicitly stated in the user's request for provided_facts.
- Put reasonable, non-factual choices needed to structure the plan in assumptions.
- Put details that are absent or unclear in missing_information.
- Put only questions that genuinely require user input in clarification_questions.
- Never invent names, dates, figures, people, technical specifications,
  commitments, or other business facts.
- Use bracketed placeholders such as [Client Name] where details are unknown.
- When essential information is missing, still produce a useful outline and ask
  specific clarification questions instead of fabricating details.
- Produce at least one section with a clear heading, purpose and key points.
- Return only a JSON object matching the schema. Do not include markdown.
"""


class PlanningError(RuntimeError):
    """Raised when a valid document plan cannot be produced."""


class DocumentPlanner:
    """Turn a normalized request into a validated document plan."""

    def __init__(self, model: BaseChatModel):
        # JSON mode avoids forced tool_choice, which is not supported in
        # DeepSeek thinking mode. It still uses Chat Completions.
        self._structured_model = model.with_structured_output(
            DocumentPlan,
            method="json_mode",
            include_raw=False,
        )

    async def create_plan(self, request: str) -> DocumentPlan:
        if not request.strip():
            raise PlanningError("Cannot create a plan for an empty request.")

        messages = [
            SystemMessage(content=PLANNING_SYSTEM_PROMPT),
            HumanMessage(content=request),
        ]

        try:
            result = await self._structured_model.ainvoke(messages)
        except Exception as exc:
            raise PlanningError(
                "The planning model could not produce a valid document plan."
            ) from exc

        if isinstance(result, DocumentPlan):
            return result

        try:
            return DocumentPlan.model_validate(result)
        except Exception as exc:
            raise PlanningError(
                "The planning model returned an invalid document plan."
            ) from exc


def build_document_planner(model: BaseChatModel | None = None) -> DocumentPlanner:
    """Build a planner, constructing the model only when needed."""
    return DocumentPlanner(model or build_chat_model())
