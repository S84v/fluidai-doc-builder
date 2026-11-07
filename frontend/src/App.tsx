import { useEffect, useRef, useState, type FormEvent } from "react";

import {
  AgentStreamError,
  streamAgentRequest,
  type ExecutionPlan,
  type PlanConflict,
  type PlanTask,
  type RenderedPlaceholder,
  type TaskStatus,
} from "./api/agent";
import { groupPlaceholders } from "./placeholderGroups";
import { cleanCompletionText, isCompletionMessage } from "./resultText";

type Phase = "idle" | "running" | "completed" | "failed";

type ExampleRequest = {
  title: string;
  description: string;
  infoLabel: string;
  prompt: string;
};

const PHASE_LABELS: Record<Phase, string> = {
  idle: "Ready",
  running: "In progress",
  completed: "Completed",
  failed: "Needs attention",
};

const TASK_STATUS_LABELS: Record<TaskStatus, string> = {
  pending: "Pending",
  active: "In progress",
  completed: "Completed",
  skipped: "Skipped",
  failed: "Failed",
};

const STAGE_LABELS: Record<string, string> = {
  request_understanding: "Request understanding",
  planning: "Planning",
  tool_selection: "Tool selection",
  tool_execution: "Tool execution",
  document_writing: "Document writing",
  validation: "Validation",
  revision: "Revision",
  docx_generation: "DOCX generation",
  finalization: "Finalization",
};

const PLAN_PREVIEW_LIMIT = 3;
const CONFLICT_PREVIEW_LIMIT = 1;

const EXAMPLE_REQUESTS: ExampleRequest[] = [
  {
    title: "Business Proposal",
    description:
      "Two-day customer service training proposal for 15 employees.",
    infoLabel: "Standard business request",
    prompt: [
      "Create a business proposal for a two-day customer service training programme for 15 employees.",
      "",
      "The intended audience is the Operations Manager. Include:",
      "- Executive summary",
      "- Objectives",
      "- Proposed training approach",
      "- Scope and deliverables",
      "- Two-day training agenda",
      "- Expected benefits",
      "- Risks and mitigation",
      "- Next steps",
      "",
      "Do not invent costs or trainer details. Use explicit placeholders where information is missing.",
    ].join("\n"),
  },
  {
    title: "Inventory System Rollout",
    description:
      "Phased inventory-system rollout across three retail branches.",
    infoLabel: "Complex business request",
    prompt: [
      "Create a detailed project plan for introducing a centralised inventory management system across a three-branch retail business.",
      "",
      "There are conflicting requirements:",
      "- Leadership wants all three branches operational within four weeks.",
      "- Operations estimates that data cleanup and staff training each require three weeks and cannot overlap.",
      "- Operations also wants a pilot in one branch before rolling out to the remaining branches.",
      "",
      "The following information is unavailable:",
      "- Project start date",
      "- Budget",
      "- Branch names",
      "- Vendor selection",
      "- Project owners and team responsibilities",
      "",
      "Analyse the requirements and decide on a sensible execution approach. Identify the conflicting constraints, dependencies and risks.",
      "",
      "Make reasonable planning assumptions where necessary, but clearly distinguish assumptions from confirmed facts. Use explicit placeholders for missing information.",
      "",
      "Propose a practical phased rollout, including milestones, responsibilities, timeline, risk mitigation and success criteria. Explain whether the requested four-week full rollout is realistic and recommend an appropriate alternative if necessary.",
    ].join("\n"),
  },
];

function formatStage(stage: string): string {
  return STAGE_LABELS[stage] ?? stage.replaceAll("_", " ");
}

function formatDocumentType(documentType: string): string {
  return documentType.replaceAll("_", " ");
}

function updateTaskStatuses(
  plan: ExecutionPlan,
  taskIds: string[],
  status: TaskStatus,
): ExecutionPlan {
  return {
    ...plan,
    tasks: plan.tasks.map((task) =>
      taskIds.includes(task.id) ? { ...task, status } : task,
    ),
  };
}

type PlanListSectionProps = {
  title: string;
  ariaLabel: string;
  items: string[];
  itemLabel: string;
  warning?: boolean;
};

function PlanListSection({
  title,
  ariaLabel,
  items,
  itemLabel,
  warning = false,
}: PlanListSectionProps) {
  if (items.length === 0) {
    return null;
  }

  const preview = items.slice(0, PLAN_PREVIEW_LIMIT);
  const remaining = items.slice(PLAN_PREVIEW_LIMIT);

  return (
    <section
      className={`detail-block${warning ? " warning-block" : ""}`}
      aria-label={ariaLabel}
    >
      <h3>{title}</h3>
      <ul>
        {preview.map((item, index) => (
          <li key={`${item}-${index}`}>{item}</li>
        ))}
      </ul>
      {remaining.length > 0 && (
        <details className="plan-disclosure">
          <summary>
            Show remaining {remaining.length}{" "}
            {remaining.length === 1 ? itemLabel : `${itemLabel}s`}
          </summary>
          <ul>
            {remaining.map((item, index) => (
              <li key={`${item}-${index + PLAN_PREVIEW_LIMIT}`}>{item}</li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}

function PlanConflictItem({ conflict }: { conflict: PlanConflict }) {
  return (
    <li>
      <ul className="nested-list">
        {conflict.requirements.map((requirement, index) => (
          <li key={`${requirement}-${index}`}>{requirement}</li>
        ))}
      </ul>
    </li>
  );
}

function PlanConflictSection({ conflicts }: { conflicts: PlanConflict[] }) {
  if (conflicts.length === 0) {
    return null;
  }

  const preview = conflicts.slice(0, CONFLICT_PREVIEW_LIMIT);
  const remaining = conflicts.slice(CONFLICT_PREVIEW_LIMIT);

  return (
    <section className="detail-block warning-block" aria-label="Conflicts">
      <h3>Conflicting requirements</h3>
      <ul>
        {preview.map((conflict, index) => (
          <PlanConflictItem
            conflict={conflict}
            key={`${conflict.requirements.join("|")}-${index}`}
          />
        ))}
      </ul>
      {remaining.length > 0 && (
        <details className="plan-disclosure">
          <summary>
            Show remaining {remaining.length}{" "}
            {remaining.length === 1 ? "conflict" : "conflicts"}
          </summary>
          <ul>
            {remaining.map((conflict, index) => (
              <PlanConflictItem
                conflict={conflict}
                key={`${conflict.requirements.join("|")}-${index + CONFLICT_PREVIEW_LIMIT}`}
              />
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}

export default function App() {
  const [request, setRequest] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [plan, setPlan] = useState<ExecutionPlan | null>(null);
  const [latestProgress, setLatestProgress] = useState("");
  const [latestStage, setLatestStage] = useState("");
  const [renderedPlaceholders, setRenderedPlaceholders] = useState<
    RenderedPlaceholder[] | null
  >(null);
  const [finalText, setFinalText] = useState("");
  const [filename, setFilename] = useState("");
  const [downloadUrl, setDownloadUrl] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  const isRunning = phase === "running";
  const canSubmit = request.trim().length > 0 && !isRunning;

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedRequest = request.trim();
    if (trimmedRequest.length === 0 || isRunning) {
      if (trimmedRequest.length === 0) {
        setPhase("failed");
        setErrorMessage("Enter a document request before generating.");
      }
      return;
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setPhase("running");
    setPlan(null);
    setLatestProgress("");
    setLatestStage("");
    setRenderedPlaceholders(null);
    setFinalText("");
    setFilename("");
    setDownloadUrl("");
    setErrorMessage("");

    try {
      await streamAgentRequest(
        trimmedRequest,
        (agentEvent) => {
          if (agentEvent.type === "plan") {
            setPlan(agentEvent.plan);
          } else if (agentEvent.type === "progress") {
            setLatestProgress(agentEvent.message);
            setLatestStage(agentEvent.stage);
            setPlan((current) =>
              current === null
                ? current
                : updateTaskStatuses(
                    current,
                    agentEvent.taskIds,
                    agentEvent.status,
                  ),
            );
          } else if (agentEvent.type === "placeholders") {
            setRenderedPlaceholders(agentEvent.placeholders);
          } else if (agentEvent.type === "metadata") {
            setPlan((current) =>
              current === null
                ? current
                : {
                    ...current,
                    documentDate: agentEvent.documentDate,
                    documentVersion: agentEvent.documentVersion,
                  },
            );
          } else if (agentEvent.type === "token") {
            setFinalText((text) => text + agentEvent.text);
          } else if (agentEvent.type === "complete") {
            setFilename(agentEvent.filename);
            setDownloadUrl(agentEvent.url);
            setPhase("completed");
          } else {
            setErrorMessage(agentEvent.message);
            setPhase("failed");
          }
        },
        controller.signal,
      );
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return;
      }

      setErrorMessage(
        error instanceof AgentStreamError
          ? error.message
          : "The document request could not be completed. Please try again.",
      );
      setPhase("failed");
    }
  }

  const placeholderList: RenderedPlaceholder[] =
    renderedPlaceholders ??
    plan?.placeholders.map((placeholder) => ({
      text: placeholder.text,
      location: placeholder.section,
      reason: placeholder.reason,
    })) ??
    [];
  const placeholderGroups = groupPlaceholders(placeholderList);
  const activeTask = plan?.tasks.find((task) => task.status === "active");
  const currentStage = activeTask?.stage ?? latestStage;
  const currentStageLabel = currentStage
    ? formatStage(currentStage)
    : "Waiting for updates";
  const completionText =
    phase === "completed"
      ? cleanCompletionText(finalText, filename)
      : finalText;
  const runningText =
    phase === "running" && !isCompletionMessage(finalText) ? finalText : "";

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header__inner">
          <p className="eyebrow">FluidAI</p>
          <h1>Doc Builder</h1>
          <p className="app-subtitle">
            Describe a business document. The agent plans it, checks it, and
            produces a Word file.
          </p>
        </div>
      </header>

      <main className="layout">
        <section className="panel request-panel" aria-labelledby="request-heading">
          <div className="panel-heading">
            <div>
              <p className="section-kicker">1. Request</p>
              <h2 id="request-heading">Document request</h2>
            </div>
          </div>

          <form onSubmit={handleSubmit} noValidate>
            <div
              className="example-grid"
              role="group"
              aria-label="Example document requests"
            >
              {EXAMPLE_REQUESTS.map((example) => (
                <div className="example-card-group" key={example.title}>
                  <button
                    type="button"
                    className="example-card"
                    onClick={() => setRequest(example.prompt)}
                    disabled={isRunning}
                    aria-label={`Use example request: ${example.title}. ${example.description}`}
                  >
                    <span className="example-card__title">{example.title}</span>
                    <span className="example-card__description">
                      {example.description}
                    </span>
                  </button>
                  <p className="example-card__info">
                    <span aria-hidden="true">ⓘ</span> {example.infoLabel}
                  </p>
                </div>
              ))}
            </div>

            <label htmlFor="document-request">What should the agent create?</label>
            <textarea
              id="document-request"
              name="request"
              value={request}
              onChange={(event) => setRequest(event.target.value)}
              placeholder="Example: Create meeting minutes for the Atlas kickoff on 6 November 2026. Attendees: Anna and Ben. Decision: proceed with phase one."
              rows={9}
              disabled={isRunning}
              aria-describedby="request-hint"
            />
            <p id="request-hint" className="field-hint">
              Include known details such as dates, attendees, decisions,
              constraints, and desired sections. Unknown details will remain
              as explicit placeholders.
            </p>

            <button type="submit" className="primary-button" disabled={!canSubmit}>
              {isRunning ? "Generating document..." : "Generate Document"}
            </button>
          </form>
        </section>

        <section
          className={`panel progress-panel phase-${phase}`}
          aria-labelledby="progress-heading"
          aria-busy={isRunning}
        >
          <div className="panel-heading">
            <div>
              <p className="section-kicker">2. Agent</p>
              <h2 id="progress-heading">Execution plan</h2>
            </div>
            <span className={`status-pill status-${phase}`}>{PHASE_LABELS[phase]}</span>
          </div>

          {plan === null && phase === "idle" && (
            <p className="empty-state">
              The generated execution plan and task progress will appear here.
            </p>
          )}
          {plan === null && phase === "running" && (
            <p className="empty-state" role="status">
              Building the request-specific execution plan...
            </p>
          )}

          {plan !== null && (
            <div className="execution-area">
              <div
                className="execution-current"
                role="status"
                aria-live="polite"
                aria-atomic="true"
              >
                <div className="execution-current__stage">
                  <span className="execution-current__label">Current stage</span>
                  <strong>{currentStageLabel}</strong>
                </div>
                <p className="execution-current__message">
                  {latestProgress ||
                    "Task updates will appear as the agent works."}
                </p>
              </div>

              <div className="task-list-heading">
                <h3>Tasks</h3>
                <span>
                  {plan.tasks.length} {plan.tasks.length === 1 ? "task" : "tasks"}
                </span>
              </div>

              {plan.tasks.length === 0 ? (
                <p className="muted-copy">
                  No tasks were returned for this plan.
                </p>
              ) : (
                <ol
                  className="task-list"
                  aria-live="polite"
                  aria-label="Execution tasks"
                >
                  {plan.tasks.map((task: PlanTask) => (
                    <li
                      key={task.id}
                      className={`task task-${task.status}`}
                      aria-current={task.status === "active" ? "step" : undefined}
                    >
                      <div className="task-heading">
                        <span className="task-title">{task.title}</span>
                        <span className={`task-status task-status-${task.status}`}>
                          {TASK_STATUS_LABELS[task.status]}
                        </span>
                      </div>
                      <p className="task-description">{task.description}</p>
                      <div className="task-meta">
                        <span className="task-stage">
                          {formatStage(task.stage)}
                        </span>
                        {task.optional && (
                          <span className="task-optional">Optional</span>
                        )}
                      </div>
                    </li>
                  ))}
                </ol>
              )}
            </div>
          )}
        </section>

        {plan !== null && (
          <section className="panel plan-panel" aria-labelledby="plan-heading">
            <div className="panel-heading">
              <div>
                <p className="section-kicker">Plan details</p>
                <h2 id="plan-heading">{plan.title}</h2>
              </div>
            </div>

            <div className="plan-summary">
              <p>
                <span>Document type</span>
                {formatDocumentType(plan.documentType)}
              </p>
              <p>
                <span>Document date</span>
                {plan.documentDate ?? "Pending generation"}
              </p>
              <p>
                <span>Document version</span>
                {plan.documentVersion ?? "Pending generation"}
              </p>
              <p className="plan-summary-objective">
                <span>Objective</span>
                {plan.objective}
              </p>
            </div>

            <div className="plan-details-grid">
              <PlanListSection
                title="Confirmed facts"
                ariaLabel="Confirmed facts"
                items={plan.providedFacts}
                itemLabel="fact"
              />
              <PlanListSection
                title="Missing information"
                ariaLabel="Missing information"
                items={plan.missingInformation}
                itemLabel="missing item"
                warning
              />
              <PlanListSection
                title="Assumptions"
                ariaLabel="Assumptions"
                items={plan.assumptions}
                itemLabel="assumption"
              />
              <PlanConflictSection conflicts={plan.conflicts} />
              <PlanListSection
                title="Questions for clarification"
                ariaLabel="Clarification questions"
                items={plan.clarificationQuestions}
                itemLabel="question"
              />

              <section className="detail-block placeholder-block" aria-label="Placeholders">
                <h3>
                  {renderedPlaceholders === null
                    ? "Planned placeholders"
                    : "Placeholders remaining in the document"}
                </h3>
                {placeholderGroups.length === 0 ? (
                  <p className="muted-copy">No unresolved placeholders remain.</p>
                ) : (
                  <div className="placeholder-scroll">
                    <ul className="placeholder-group-list">
                      {placeholderGroups.map((group) => (
                        <li key={JSON.stringify([group.text, group.reason])}>
                          <div className="placeholder-heading">
                            <code>{group.text}</code>
                            <span className="placeholder-count">
                              {group.count}{" "}
                              {group.count === 1 ? "occurrence" : "occurrences"}
                            </span>
                          </div>
                          <p>{group.reason}</p>
                          {group.locations.length > 1 ? (
                            <details className="placeholder-locations">
                              <summary>
                                {group.locations.length} locations
                              </summary>
                              <ul>
                                {group.locations.map((location) => (
                                  <li key={location}>{location}</li>
                                ))}
                              </ul>
                            </details>
                          ) : (
                            <span className="placeholder-location">
                              {group.locations[0]}
                            </span>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </section>
            </div>
          </section>
        )}

        <section
          className={`panel result-panel result-${phase}`}
          aria-labelledby="result-heading"
        >
          <div className="panel-heading">
            <div>
              <p className="section-kicker">3. Result</p>
              <h2 id="result-heading">Generated document</h2>
            </div>
          </div>

          {phase === "idle" && (
            <p className="empty-state">
              The completed document filename and download link will appear
              here.
            </p>
          )}

          {phase === "running" && runningText.length === 0 && (
            <p className="empty-state" role="status">
              The agent is working on your request.
            </p>
          )}

          {phase === "running" && runningText.length > 0 && (
            <p className="final-text">{runningText}</p>
          )}

          {phase === "completed" && downloadUrl.length > 0 && (
            <div className="completion-block">
              <div className="completion-status" role="status">
                <span className="completion-status__icon" aria-hidden="true">
                  ✓
                </span>
                <div>
                  <p className="completion-status__title">Generation complete</p>
                  <p className="completion-status__copy">
                    Your document is ready to download.
                  </p>
                </div>
              </div>

              {completionText.length > 0 && (
                <p className="final-text">{completionText}</p>
              )}

              <div className="completion-file">
                <p className="completion-label">Filename</p>
                <code className="filename">{filename}</code>
              </div>

              <a className="download-link" href={downloadUrl} download={filename}>
                Download Document
              </a>
            </div>
          )}

          {phase === "failed" && errorMessage.length > 0 && (
            <div className="error-box" role="alert">
              <strong>The document was not generated.</strong>
              <p>{errorMessage}</p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
