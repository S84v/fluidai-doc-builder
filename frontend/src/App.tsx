import { useEffect, useRef, useState, type FormEvent } from "react";

import {
  AgentStreamError,
  streamAgentRequest,
  type ExecutionPlan,
  type PlanTask,
  type RenderedPlaceholder,
  type TaskStatus,
} from "./api/agent";

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

export default function App() {
  const [request, setRequest] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [plan, setPlan] = useState<ExecutionPlan | null>(null);
  const [latestProgress, setLatestProgress] = useState("");
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

  const placeholderList =
    renderedPlaceholders ??
    plan?.placeholders.map((placeholder) => ({
      text: placeholder.text,
      location: placeholder.section,
      reason: placeholder.reason,
    })) ??
    [];

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
            <>
              {latestProgress && (
                <p className="latest-progress" role="status">
                  {latestProgress}
                </p>
              )}
              <ol className="task-list" aria-live="polite">
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
                      <span>{formatStage(task.stage)}</span>
                      {task.optional && <span>Optional</span>}
                    </div>
                  </li>
                ))}
              </ol>
            </>
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
              <p>
                <span>Objective</span>
                {plan.objective}
              </p>
            </div>

            <div className="plan-details-grid">
              {plan.conflicts.length > 0 && (
                <section className="detail-block warning-block" aria-label="Conflicts">
                  <h3>Conflicting requirements</h3>
                  <ul>
                    {plan.conflicts.map((conflict, index) => (
                      <li key={`${index}-${conflict.impact}`}>
                        <ul className="nested-list">
                          {conflict.requirements.map((requirement) => (
                            <li key={requirement}>{requirement}</li>
                          ))}
                        </ul>
                        <p>
                          <strong>Impact:</strong> {conflict.impact}
                        </p>
                        <p>
                          <strong>Resolution:</strong>{" "}
                          {conflict.resolution ?? "Unresolved"}
                        </p>
                      </li>
                    ))}
                  </ul>
                </section>
              )}

              {plan.missingInformation.length > 0 && (
                <section
                  className="detail-block warning-block"
                  aria-label="Missing information"
                >
                  <h3>Missing information</h3>
                  <ul>
                    {plan.missingInformation.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </section>
              )}

              {plan.assumptions.length > 0 && (
                <section className="detail-block" aria-label="Assumptions">
                  <h3>Assumptions</h3>
                  <ul>
                    {plan.assumptions.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </section>
              )}

              {plan.providedFacts.length > 0 && (
                <section className="detail-block" aria-label="Confirmed facts">
                  <h3>Confirmed facts</h3>
                  <ul>
                    {plan.providedFacts.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </section>
              )}

              {plan.clarificationQuestions.length > 0 && (
                <section
                  className="detail-block"
                  aria-label="Clarification questions"
                >
                  <h3>Questions for clarification</h3>
                  <ul>
                    {plan.clarificationQuestions.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </section>
              )}

              <section className="detail-block placeholder-block" aria-label="Placeholders">
                <h3>
                  {renderedPlaceholders === null
                    ? "Planned placeholders"
                    : "Placeholders remaining in the document"}
                </h3>
                {placeholderList.length === 0 ? (
                  <p className="muted-copy">No unresolved placeholders remain.</p>
                ) : (
                  <ul>
                    {placeholderList.map((placeholder, index) => (
                      <li key={`${placeholder.text}-${placeholder.location}-${index}`}>
                        <code>{placeholder.text}</code>
                        <span className="placeholder-location">
                          {placeholder.location}
                        </span>
                        <p>{placeholder.reason}</p>
                      </li>
                    ))}
                  </ul>
                )}
              </section>
            </div>
          </section>
        )}

        <section className="panel result-panel" aria-labelledby="result-heading">
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

          {phase === "running" && finalText.length === 0 && (
            <p className="empty-state" role="status">
              The agent is working on your request.
            </p>
          )}

          {finalText.length > 0 && <p className="final-text">{finalText}</p>}

          {phase === "completed" && downloadUrl.length > 0 && (
            <div className="completion-block">
              <p className="completion-label">Generated file</p>
              <code className="filename">{filename}</code>
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
