// // web/src/App.jsx
// // React front end for the Agentic Document QA system.
// // It talks to server.py through a small JSON API (/api/...).
// // Build (only needed if you edit this file): npm install && npm run build

// import React, { useCallback, useEffect, useRef, useState } from "react";
// import { createRoot } from "react-dom/client";

// const MODES = [
//   { id: "agentic", label: "Agentic", hint: "can search again" },
//   { id: "single", label: "Single pass", hint: "one search, one answer" },
//   { id: "retrieval", label: "Retrieval only", hint: "no LLM, no API key needed" },
// ];

// // Edit these to match your own documents.
// const EXAMPLES = [
//   "how does the CYK algorithm parse a sentence?",
//   "what are its limitations?",
//   "what is the capital of France?",
// ];

// // const LEVEL_LABEL = { high: "High", medium: "Medium", low: "Low" };
// const LEVEL_LABEL = {
//   high: "High retrieval similarity",
//   medium: "Medium retrieval similarity",
//   low: "Low retrieval similarity",
// };

// async function api(path, options) {
//   const res = await fetch(path, options);
//   if (!res.ok) {
//     const body = await res.text();
//     throw new Error(`${res.status} ${body.slice(0, 200)}`);
//   }
//   return res.json();
// }

// /* ------------------------------------------------------------ text helpers */

// // Turns **bold** into <strong> and [1] citation markers into small superscripts.
// // function renderInline(text) {
// //   return text.split(/(\*\*[^*]+\*\*|\[\d+\])/g).map((part, i) => {
// //     if (/^\*\*[^*]+\*\*$/.test(part)) return <strong key={i}>{part.slice(2, -2)}</strong>;
// //     if (/^\[\d+\]$/.test(part)) return <sup key={i} className="cite">{part.slice(1, -1)}</sup>;
// //     return part;
// //   });
// // }


// // function renderInline(text) {
// //   const parts = text.split(/(\[\d+\]|\*\*)/g);
// //   const nodes = [];

// //   let bold = false;
// //   let previousWasCitation = false;

// //   parts.forEach((part, i) => {
// //     if (part === "**") {
// //       bold = !bold;
// //       previousWasCitation = false;
// //       return;
// //     }

// //     if (/^\[\d+\]$/.test(part)) {
// //       // Separate adjacent citations for better readability.
// //       if (previousWasCitation) {
// //         nodes.push(", ");
// //       }

// //       nodes.push(
// //         <sup key={`cite-${i}`} className="cite">
// //           {part.slice(1, -1)}
// //         </sup>
// //       );

// //       previousWasCitation = true;
// //       return;
// //     }

// //     if (part) {
// //       nodes.push(
// //         bold ? <strong key={`bold-${i}`}>{part}</strong> : part
// //       );
// //       previousWasCitation = false;
// //     }
// //   });

// //   return nodes;
// // }

// function renderInline(text) {
//   const parts = String(text ?? "").split(
//     /(\[\d+\]|\*\*|`[^`]+`|\*)/g
//   );

//   const nodes = [];
//   let bold = false;
//   let italic = false;
//   let previousWasCitation = false;

//   parts.forEach((part, i) => {
//     if (part === "**") {
//       bold = !bold;
//       return;
//     }

//     if (part === "*") {
//       italic = !italic;
//       return;
//     }

//     if (/^\[\d+\]$/.test(part)) {
//       if (previousWasCitation) {
//         nodes.push(", ");
//       }

//       nodes.push(
//         <sup key={`cite-${i}`} className="cite">
//           {part.slice(1, -1)}
//         </sup>
//       );

//       previousWasCitation = true;
//       return;
//     }

//     if (/^`[^`]+`$/.test(part)) {
//       nodes.push(
//         <code key={`code-${i}`}>
//           {part.slice(1, -1)}
//         </code>
//       );
//       previousWasCitation = false;
//       return;
//     }

//     if (part) {
//       if (bold && italic) {
//         nodes.push(<strong key={i}><em>{part}</em></strong>);
//       } else if (bold) {
//         nodes.push(<strong key={i}>{part}</strong>);
//       } else if (italic) {
//         nodes.push(<em key={i}>{part}</em>);
//       } else {
//         nodes.push(part);
//       }

//       previousWasCitation = false;
//     }
//   });

//   return nodes;
// }

// // // Minimal Markdown: paragraphs and bullet / numbered lists. Nothing is injected as HTML.
// // function RichText({ text }) {
// //   const blocks = [];
// //   let list = null;
// //   text.split("\n").forEach((line) => {
// //     const m = line.match(/^\s*(?:[-*\u2022]|\d+\.)\s+(.*)$/);
// //     if (m) {
// //       if (!list) {
// //         list = [];
// //         blocks.push({ type: "ul", items: list });
// //       }
// //       list.push(m[1]);
// //     } else {
// //       list = null;
// //       if (line.trim()) blocks.push({ type: "p", text: line });
// //     }
// //   });
// //   return (
// //     <div className="rich">
// //       {blocks.map((b, i) =>
// //         b.type === "ul" ? (
// //           <ul key={i}>
// //             {b.items.map((it, j) => (
// //               <li key={j}>{renderInline(it)}</li>
// //             ))}
// //           </ul>
// //         ) : (
// //           <p key={i}>{renderInline(b.text)}</p>
// //         )
// //       )}
// //     </div>
// //   );
// // }

// function RichText({ text }) {
//   const lines = String(text ?? "")
//     .replace(/\r\n?/g, "\n")
//     .split("\n");

//   const blocks = [];
//   let activeList = null;

//   const nextNonEmpty = (start) => {
//     let index = start;

//     while (index < lines.length && !lines[index].trim()) {
//       index++;
//     }

//     return index;
//   };

//   const splitTableRow = (line) => {
//     return line
//       .trim()
//       .replace(/\\\|/g, "|")
//       .replace(/^\|/, "")
//       .replace(/\|$/, "")
//       .split("|")
//       .map((cell) => cell.trim());
//   };

//   const isTableSeparator = (line) => {
//     const cells = splitTableRow(line);

//     return (
//       cells.length > 1 &&
//       cells.every((cell) => /^:?-{3,}:?$/.test(cell))
//     );
//   };

//   for (let i = 0; i < lines.length; i++) {
//     const line = lines[i];

//     if (!line.trim()) {
//       activeList = null;
//       continue;
//     }

//     // Markdown headings: # through ######
//     const heading = line.match(
//       /^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/
//     );

//     if (heading) {
//       activeList = null;

//       blocks.push({
//         type: "heading",
//         level: heading[1].length,
//         text: heading[2],
//       });

//       continue;
//     }

//     // Horizontal separator
//     if (/^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
//       activeList = null;
//       blocks.push({ type: "hr" });
//       continue;
//     }

//     // Markdown tables
//     if (line.includes("|")) {
//       const separatorIndex = nextNonEmpty(i + 1);

//       if (
//         separatorIndex < lines.length &&
//         isTableSeparator(lines[separatorIndex])
//       ) {
//         activeList = null;

//         const headers = splitTableRow(line);
//         const separators = splitTableRow(lines[separatorIndex]);

//         const alignments = separators.map((cell) => {
//           if (cell.startsWith(":") && cell.endsWith(":")) {
//             return "center";
//           }

//           if (cell.endsWith(":")) {
//             return "right";
//           }

//           return "left";
//         });

//         const rows = [];
//         let rowIndex = separatorIndex + 1;

//         while (rowIndex < lines.length) {
//           if (!lines[rowIndex].trim()) {
//             const nextRow = nextNonEmpty(rowIndex + 1);

//             if (
//               nextRow < lines.length &&
//               lines[nextRow].includes("|")
//             ) {
//               rowIndex = nextRow;
//             } else {
//               break;
//             }
//           }

//           if (!lines[rowIndex].includes("|")) {
//             break;
//           }

//           rows.push(splitTableRow(lines[rowIndex]));
//           rowIndex++;
//         }

//         blocks.push({
//           type: "table",
//           headers,
//           rows,
//           alignments,
//         });

//         i = rowIndex - 1;
//         continue;
//       }
//     }

//     // Unordered and ordered lists
//     const unordered = line.match(/^\s*[-+*]\s+(.*)$/);
//     const ordered = line.match(/^\s*\d+\.\s+(.*)$/);

//     if (unordered || ordered) {
//       const type = unordered ? "ul" : "ol";
//       const item = unordered ? unordered[1] : ordered[1];

//       if (!activeList || activeList.type !== type) {
//         activeList = { type, items: [] };
//         blocks.push(activeList);
//       }

//       activeList.items.push(item);
//       continue;
//     }

//     // Normal paragraph
//     activeList = null;
//     blocks.push({ type: "p", text: line });
//   }

//   return (
//     <div className="rich">
//       {blocks.map((b, i) => {
//         if (b.type === "heading") {
//           const HeadingTag = `h${b.level}`;

//           return (
//             <HeadingTag key={i}>
//               {renderInline(b.text)}
//             </HeadingTag>
//           );
//         }

//         if (b.type === "hr") {
//           return <hr key={i} className="rich-divider" />;
//         }

//         if (b.type === "ul" || b.type === "ol") {
//           const ListTag = b.type;

//           return (
//             <ListTag key={i}>
//               {b.items.map((item, j) => (
//                 <li key={j}>{renderInline(item)}</li>
//               ))}
//             </ListTag>
//           );
//         }

//         if (b.type === "table") {
//           return (
//             <div className="rich-table-wrap" key={i}>
//               <table className="rich-table">
//                 <thead>
//                   <tr>
//                     {b.headers.map((cell, j) => (
//                       <th
//                         key={j}
//                         style={{
//                           textAlign: b.alignments[j] || "left",
//                         }}
//                       >
//                         {renderInline(cell)}
//                       </th>
//                     ))}
//                   </tr>
//                 </thead>

//                 <tbody>
//                   {b.rows.map((row, j) => (
//                     <tr key={j}>
//                       {b.headers.map((header, k) => (
//                         <td
//                           key={k}
//                           style={{
//                             textAlign: b.alignments[k] || "left",
//                           }}
//                         >
//                           {renderInline(row[k] || "")}
//                         </td>
//                       ))}
//                     </tr>
//                   ))}
//                 </tbody>
//               </table>
//             </div>
//           );
//         }

//         return (
//           <p key={i}>
//             {renderInline(b.text)}
//           </p>
//         );
//       })}
//     </div>
//   );
// }

// /* --------------------------------------------------------------- messages */

// // function ConfidenceBadge({ level, score }) {
// //   return (
// //     <span
// //       className={`badge badge-${level}`}
// //       title="Based on the best similarity score of the retrieved passages"
// //     >
// //       {LEVEL_LABEL[level] || level} confidence &middot; {score.toFixed(2)}
// //     </span>
// //   );
// // }

// function ConfidenceBadge({ level, score }) {
//   return (
//     <span
//       className={`badge badge-${level}`}
//       title="Based on the highest dense similarity score. This does not establish whether the retrieved passages contain the answer."
//     >
//       {LEVEL_LABEL[level] || level} &middot; {score.toFixed(2)}
//     </span>
//   );
// }

// function AssistantMessage({ msg }) {
//   if (msg.kind === "error") {
//     return (
//       <div className="notice notice-error pre" role="alert">
//         {msg.text}
//       </div>
//     );
//   }

//   const isRetrieval = msg.kind === "retrieval";
//   return (
//     <div className="assistant">
//       <div className="meta">
//         <ConfidenceBadge level={msg.level} score={msg.score} />
//         {!isRetrieval && <span className="meta-item">LLM calls: {msg.calls}</span>}
//         {isRetrieval && <span className="meta-item">Retrieval only (no LLM)</span>}
//       </div>

//       {isRetrieval ? (
//         <>
//           {msg.level !== "high" && msg.message && <div className="notice notice-info">{msg.message}</div>}
//           {msg.chunks.length === 0 && <p className="muted">No passages found.</p>}
//           {msg.chunks.map((c, i) => (
//             <details className="card" key={i}>
//               <summary>
//                 <span className="card-title">
//                   {i + 1}. {c.doc}, p.{c.page}
//                 </span>
//                 <span className="chip">similarity {c.dense.toFixed(2)}</span>
//               </summary>
//               <p className="pre card-body">{c.text}</p>
//             </details>
//           ))}
//         </>
//       ) : (
//         <>
//           {msg.standalone !== msg.original && (
//             <p className="muted small">Understood as: {msg.standalone}</p>
//           )}
//           {msg.level === "medium" && msg.message && (
//             <div className="notice notice-warn">{msg.message}</div>
//           )}
//           <RichText text={msg.text} />

          
//           {msg.sources.length > 0 && (
//             <section className="block">
//               <h4>Sources ({msg.sources.length})</h4>

//               {msg.sources.map((s) => (
//                 <details className="source" key={s.n}>
//                   <summary>
//                   <span className="source-n">{s.n}</span>

//                   <span>
//                     <span className="source-title">
//                       {s.doc}, p.{s.page}
//                     </span>

//                     <span className="muted small">
//                       &ldquo;{s.snippet}&hellip;&rdquo;
//                     </span>
//                   </span>

//                   <span className="muted small">View passage</span>
//                 </summary>

//                 <div className="pre card-body">
//                   {s.full_text || s.snippet}
//                 </div>
//               </details>
//             ))}
//           </section>
//           )}

          

//           {msg.trace.length > 0 && (
//             <details className="block">
//               <summary>How the agent searched</summary>
//               <ol className="trace">
//                 {msg.trace.map((t, i) => (
//                   <li key={i}>
//                     <code>{t.query}</code> &rarr; +{t.new_chunks} new passages &rarr; <strong>{t.decision}</strong>
//                   </li>
//                 ))}
//               </ol>
//             </details>
//           )}
//         </>
//       )}
//     </div>
//   );
// }

// function Message({ msg }) {
//   if (msg.role === "user") {
//     return (
//       <div className="row row-user">
//         <div className="bubble bubble-user pre">{msg.text}</div>
//       </div>
//     );
//   }
//   return (
//     <div className="row">
//       <div className="avatar" aria-hidden="true">AI</div>
//       <div className="bubble bubble-ai">
//         <AssistantMessage msg={msg} />
//       </div>
//     </div>
//   );
// }

// /* -------------------------------------------------------------------- app */

// function App() {
//   const [status, setStatus] = useState({ documents: [], llm: { connected: false, name: null }, default_model: "" });
//   const [messages, setMessages] = useState([]);
//   const [input, setInput] = useState("");
//   const [mode, setMode] = useState("agentic");
//   const [selected, setSelected] = useState([]);
//   const [busy, setBusy] = useState(false);
//   const [notice, setNotice] = useState(null);
//   const [apiKey, setApiKey] = useState("");
//   const [model, setModel] = useState("");
//   const [uploading, setUploading] = useState(false);
//   const [dragging, setDragging] = useState(false);
//   const endRef = useRef(null);
//   const fileRef = useRef(null);

//   useEffect(() => {
//     api("/api/status")
//       .then((s) => {
//         setStatus(s);
//         setModel((m) => m || s.default_model);
//       })
//       .catch((e) => setNotice({ type: "error", text: `Cannot reach the server: ${e.message}` }));
//   }, []);

//   useEffect(() => {
//     const el = endRef.current;
//     if (el && el.scrollIntoView) el.scrollIntoView({ behavior: "smooth", block: "end" });
//   }, [messages, busy]);

//   const send = useCallback(
//     async (text) => {
//       const question = text.trim();
//       if (!question || busy) return;
//       setMessages((m) => [...m, { role: "user", text: question }]);
//       setInput("");
//       setBusy(true);
//       try {
//         const data = await api("/api/ask", {
//           method: "POST",
//           headers: { "Content-Type": "application/json" },
//           body: JSON.stringify({ question, mode, docs: selected.length ? selected : null }),
//         });
//         setMessages((m) => [...m, { role: "assistant", ...data }]);
//       } catch (e) {
//         setMessages((m) => [
//           ...m,
//           { role: "assistant", kind: "error", text: `Could not reach the server: ${e.message}` },
//         ]);
//       } finally {
//         setBusy(false);
//       }
//     },
//     [busy, mode, selected]
//   );

//   async function uploadFiles(fileList) {
//     const files = Array.from(fileList || []);
//     if (!files.length) return;
//     const form = new FormData();
//     files.forEach((f) => form.append("files", f));
//     setUploading(true);
//     setNotice(null);
//     try {
//       const data = await api("/api/upload", { method: "POST", body: form });
//       setStatus(data);
//       const problems = data.results.filter((r) => r.note);
//       setNotice(
//         problems.length
//           ? { type: "warn", text: problems.map((r) => `${r.name}: ${r.note}`).join(" ") }
//           : { type: "ok", text: `Loaded ${data.results.map((r) => r.name).join(", ")}.` }
//       );
//     } catch (e) {
//       setNotice({ type: "error", text: `Upload failed: ${e.message}` });
//     } finally {
//       setUploading(false);
//       if (fileRef.current) fileRef.current.value = "";
//     }
//   }

//   async function loadFolder() {
//     setNotice(null);
//     try {
//       const data = await api("/api/load-folder", { method: "POST" });
//       setStatus(data);
//       setNotice({ type: "ok", text: "Loaded the PDFs from data/pdfs." });
//     } catch (e) {
//       setNotice({ type: "error", text: `Could not load the folder: ${e.message}` });
//     }
//   }

//   async function connect() {
//     setNotice(null);
//     try {
//       const data = await api("/api/connect", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ api_key: apiKey, model }),
//       });
//       setStatus(data);
//       setNotice({ type: data.ok ? "ok" : "error", text: data.ok ? `Connected: ${data.info}` : data.info });
//       if (data.ok) setApiKey("");
//     } catch (e) {
//       setNotice({ type: "error", text: `Could not connect: ${e.message}` });
//     }
//   }

//   async function clearChat() {
//     try {
//       await api("/api/reset", { method: "POST" });
//     } catch (e) {
//       /* ignore: the page is cleared either way */
//     }
//     setMessages([]);
//   }

//   function toggleDoc(name) {
//     setSelected((cur) => (cur.includes(name) ? cur.filter((d) => d !== name) : [...cur, name]));
//   }

//   const noDocs = status.documents.length === 0;

//   return (
//     <div className="app">
//       <aside className="sidebar">
//         <h1 className="brand">Document QA</h1>

//         <section>
//           <h2>Documents</h2>
//           <div
//             className={`drop ${dragging ? "drop-active" : ""}`}
//             onDragOver={(e) => {
//               e.preventDefault();
//               setDragging(true);
//             }}
//             onDragLeave={() => setDragging(false)}
//             onDrop={(e) => {
//               e.preventDefault();
//               setDragging(false);
//               uploadFiles(e.dataTransfer.files);
//             }}
//           >
//             <p>{uploading ? "Reading PDFs..." : "Drop PDFs here or"}</p>
//             <label className="btn btn-secondary">
//               Choose files
//               <input
//                 ref={fileRef}
//                 type="file"
//                 accept="application/pdf,.pdf"
//                 multiple
//                 hidden
//                 onChange={(e) => uploadFiles(e.target.files)}
//               />
//             </label>
//           </div>
//           <button className="btn btn-ghost" onClick={loadFolder}>
//             Load PDFs from data/pdfs
//           </button>

//           {noDocs ? (
//             <p className="muted small">No documents loaded yet.</p>
//           ) : (
//             <ul className="docs">
//               {status.documents.map((d) => (
//                 <li key={d.name}>
//                   <label title="Tick to restrict answers to the ticked documents">
//                     <input
//                       type="checkbox"
//                       checked={selected.includes(d.name)}
//                       onChange={() => toggleDoc(d.name)}
//                     />
//                     <span className="doc-name">{d.name}</span>
//                     <span className="chip">{d.passages}</span>
//                   </label>
//                 </li>
//               ))}
//             </ul>
//           )}
//           {!noDocs && (
//             <p className="muted small">
//               {selected.length ? `Answers use only the ${selected.length} ticked document(s).` : "Answers use all documents."}
//             </p>
//           )}
//         </section>

//         <section>
//           <h2>Mode</h2>
//           {MODES.map((m) => (
//             <label className="radio" key={m.id}>
//               <input type="radio" name="mode" checked={mode === m.id} onChange={() => setMode(m.id)} />
//               <span>
//                 <strong>{m.label}</strong>
//                 <span className="muted small"> {m.hint}</span>
//               </span>
//             </label>
//           ))}
//         </section>

//         <section>
//           <h2>Gemini</h2>
//           <p className="status">
//             <span className={`dot ${status.llm.connected ? "dot-on" : ""}`} />
//             {status.llm.connected ? `Connected: ${status.llm.name}` : "Not connected"}
//           </p>
//           <label className="field">
//             API key
//             <input
//               type="password"
//               autoComplete="off"
//               placeholder="or set GEMINI_API_KEY before starting"
//               value={apiKey}
//               onChange={(e) => setApiKey(e.target.value)}
//             />
//           </label>
//           <label className="field">
//             Model
//             <input type="text" value={model} onChange={(e) => setModel(e.target.value)} />
//           </label>
//           <button className="btn btn-secondary" onClick={connect}>
//             Connect Gemini
//           </button>
//           <p className="muted small">The key is sent only to this local server and is not saved to disk.</p>
//         </section>

//         <section>
//           <h2>Conversation</h2>
//           <button className="btn btn-ghost" onClick={clearChat}>
//             Clear conversation
//           </button>
//           <h3>Examples</h3>
//           <div className="examples">
//             {EXAMPLES.map((ex) => (
//               <button key={ex} className="chip chip-button" onClick={() => send(ex)} disabled={busy}>
//                 {ex}
//               </button>
//             ))}
//           </div>
//         </section>
//       </aside>

//       <main className="main">
//         {notice && (
//           <div className={`notice notice-${notice.type} pre`} role="status">
//             {notice.text}
//             <button className="close" aria-label="Dismiss" onClick={() => setNotice(null)}>
//               &times;
//             </button>
//           </div>
//         )}

//         <div className="messages" role="log" aria-live="polite">
//           {messages.length === 0 && (
//             <div className="empty">
//               <h2>Ask a question about your documents</h2>
//               {/* <p className="muted">
//                 Answers show their sources (file and page) and a confidence level. Follow-up questions such as
//                 &ldquo;what are its limitations?&rdquo; are understood from the conversation.
//               </p> */}
//               <p className="muted">
//                 Answers show their sources (file and page), along with the retrieval similarity of the
//                 passages found. Retrieval similarity indicates how closely passages match your question;
//                 it does not guarantee that they contain the answer. Follow-up questions such as
//                 &ldquo;what are its limitations?&rdquo; are understood from the conversation.
//               </p>
//               {noDocs && <div className="notice notice-info">Upload one or more PDFs in the sidebar to begin.</div>}
//             </div>
//           )}

//           {messages.map((m, i) => (
//             <Message key={i} msg={m} />
//           ))}

//           {busy && (
//             <div className="row">
//               <div className="avatar" aria-hidden="true">AI</div>
//               <div className="bubble bubble-ai">
//                 <span className="typing" aria-label="Working">
//                   <i /> <i /> <i />
//                 </span>
//                 {mode !== "retrieval" && (
//                   <p className="muted small">Free Gemini limits can make this take a minute or two.</p>
//                 )}
//               </div>
//             </div>
//           )}
//           <div ref={endRef} />
//         </div>

//         <form
//           className="composer"
//           onSubmit={(e) => {
//             e.preventDefault();
//             send(input);
//           }}
//         >
//           <textarea
//             rows={2}
//             placeholder="Ask a question about your documents (Enter to send, Shift+Enter for a new line)"
//             value={input}
//             onChange={(e) => setInput(e.target.value)}
//             onKeyDown={(e) => {
//               if (e.key === "Enter" && !e.shiftKey) {
//                 e.preventDefault();
//                 send(input);
//               }
//             }}
//           />
//           <button className="btn btn-primary" type="submit" disabled={busy || !input.trim()}>
//             Send
//           </button>
//         </form>
//       </main>
//     </div>
//   );
// }

// createRoot(document.getElementById("root")).render(<App />);













// web/src/App.jsx
// React front end for the Agentic Document QA system.
// It talks to server.py through a small JSON API (/api/...).
// Build (only needed if you edit this file): npm install && npm run build

import React, { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

const MODES = [
  { id: "agentic", label: "Agentic", hint: "can search again" },
  { id: "single", label: "Single pass", hint: "one search, one answer" },
  { id: "retrieval", label: "Retrieval only", hint: "no LLM, no API key needed" },
];

// Edit these to match your own documents.
const EXAMPLES = [
  "how does the CYK algorithm parse a sentence?",
  "what are its limitations?",
  "what is the capital of France?",
];

// const LEVEL_LABEL = { high: "High", medium: "Medium", low: "Low" };
const LEVEL_LABEL = {
  high: "High retrieval similarity",
  medium: "Medium retrieval similarity",
  low: "Low retrieval similarity",
};

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${body.slice(0, 200)}`);
  }
  return res.json();
}

/* ------------------------------------------------------------ text helpers */

// Turns **bold** into <strong> and [1] citation markers into small superscripts.
// function renderInline(text) {
//   return text.split(/(\*\*[^*]+\*\*|\[\d+\])/g).map((part, i) => {
//     if (/^\*\*[^*]+\*\*$/.test(part)) return <strong key={i}>{part.slice(2, -2)}</strong>;
//     if (/^\[\d+\]$/.test(part)) return <sup key={i} className="cite">{part.slice(1, -1)}</sup>;
//     return part;
//   });
// }


// function renderInline(text) {
//   const parts = text.split(/(\[\d+\]|\*\*)/g);
//   const nodes = [];

//   let bold = false;
//   let previousWasCitation = false;

//   parts.forEach((part, i) => {
//     if (part === "**") {
//       bold = !bold;
//       previousWasCitation = false;
//       return;
//     }

//     if (/^\[\d+\]$/.test(part)) {
//       // Separate adjacent citations for better readability.
//       if (previousWasCitation) {
//         nodes.push(", ");
//       }

//       nodes.push(
//         <sup key={`cite-${i}`} className="cite">
//           {part.slice(1, -1)}
//         </sup>
//       );

//       previousWasCitation = true;
//       return;
//     }

//     if (part) {
//       nodes.push(
//         bold ? <strong key={`bold-${i}`}>{part}</strong> : part
//       );
//       previousWasCitation = false;
//     }
//   });

//   return nodes;
// }

function renderInline(text) {
  const parts = String(text ?? "").split(
    /(\[\d+\]|\*\*|`[^`]+`|\*)/g
  );

  const nodes = [];
  let bold = false;
  let italic = false;
  let previousWasCitation = false;

  parts.forEach((part, i) => {
    if (part === "**") {
      bold = !bold;
      return;
    }

    if (part === "*") {
      italic = !italic;
      return;
    }

    if (/^\[\d+\]$/.test(part)) {
      if (previousWasCitation) {
        nodes.push(", ");
      }

      nodes.push(
        <sup key={`cite-${i}`} className="cite">
          {part.slice(1, -1)}
        </sup>
      );

      previousWasCitation = true;
      return;
    }

    if (/^`[^`]+`$/.test(part)) {
      nodes.push(
        <code key={`code-${i}`}>
          {part.slice(1, -1)}
        </code>
      );
      previousWasCitation = false;
      return;
    }

    if (part) {
      if (bold && italic) {
        nodes.push(<strong key={i}><em>{part}</em></strong>);
      } else if (bold) {
        nodes.push(<strong key={i}>{part}</strong>);
      } else if (italic) {
        nodes.push(<em key={i}>{part}</em>);
      } else {
        nodes.push(part);
      }

      previousWasCitation = false;
    }
  });

  return nodes;
}

// // Minimal Markdown: paragraphs and bullet / numbered lists. Nothing is injected as HTML.
// function RichText({ text }) {
//   const blocks = [];
//   let list = null;
//   text.split("\n").forEach((line) => {
//     const m = line.match(/^\s*(?:[-*\u2022]|\d+\.)\s+(.*)$/);
//     if (m) {
//       if (!list) {
//         list = [];
//         blocks.push({ type: "ul", items: list });
//       }
//       list.push(m[1]);
//     } else {
//       list = null;
//       if (line.trim()) blocks.push({ type: "p", text: line });
//     }
//   });
//   return (
//     <div className="rich">
//       {blocks.map((b, i) =>
//         b.type === "ul" ? (
//           <ul key={i}>
//             {b.items.map((it, j) => (
//               <li key={j}>{renderInline(it)}</li>
//             ))}
//           </ul>
//         ) : (
//           <p key={i}>{renderInline(b.text)}</p>
//         )
//       )}
//     </div>
//   );
// }

function RichText({ text }) {
  const lines = String(text ?? "")
    .replace(/\r\n?/g, "\n")
    .split("\n");

  const blocks = [];
  let activeList = null;

  const nextNonEmpty = (start) => {
    let index = start;

    while (index < lines.length && !lines[index].trim()) {
      index++;
    }

    return index;
  };

  const splitTableRow = (line) => {
    return line
      .trim()
      .replace(/\\\|/g, "|")
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((cell) => cell.trim());
  };

  const isTableSeparator = (line) => {
    const cells = splitTableRow(line);

    return (
      cells.length > 1 &&
      cells.every((cell) => /^:?-{3,}:?$/.test(cell))
    );
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (!line.trim()) {
      activeList = null;
      continue;
    }

    // Markdown headings: # through ######
    const heading = line.match(
      /^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/
    );

    if (heading) {
      activeList = null;

      blocks.push({
        type: "heading",
        level: heading[1].length,
        text: heading[2],
      });

      continue;
    }

    // Horizontal separator
    if (/^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      activeList = null;
      blocks.push({ type: "hr" });
      continue;
    }

    // Markdown tables
    if (line.includes("|")) {
      const separatorIndex = nextNonEmpty(i + 1);

      if (
        separatorIndex < lines.length &&
        isTableSeparator(lines[separatorIndex])
      ) {
        activeList = null;

        const headers = splitTableRow(line);
        const separators = splitTableRow(lines[separatorIndex]);

        const alignments = separators.map((cell) => {
          if (cell.startsWith(":") && cell.endsWith(":")) {
            return "center";
          }

          if (cell.endsWith(":")) {
            return "right";
          }

          return "left";
        });

        const rows = [];
        let rowIndex = separatorIndex + 1;

        while (rowIndex < lines.length) {
          if (!lines[rowIndex].trim()) {
            const nextRow = nextNonEmpty(rowIndex + 1);

            if (
              nextRow < lines.length &&
              lines[nextRow].includes("|")
            ) {
              rowIndex = nextRow;
            } else {
              break;
            }
          }

          if (!lines[rowIndex].includes("|")) {
            break;
          }

          rows.push(splitTableRow(lines[rowIndex]));
          rowIndex++;
        }

        blocks.push({
          type: "table",
          headers,
          rows,
          alignments,
        });

        i = rowIndex - 1;
        continue;
      }
    }

    // Unordered and ordered lists
    const unordered = line.match(/^\s*[-+*]\s+(.*)$/);
    const ordered = line.match(/^\s*\d+\.\s+(.*)$/);

    if (unordered || ordered) {
      const type = unordered ? "ul" : "ol";
      const item = unordered ? unordered[1] : ordered[1];

      if (!activeList || activeList.type !== type) {
        activeList = { type, items: [] };
        blocks.push(activeList);
      }

      activeList.items.push(item);
      continue;
    }

    // Normal paragraph
    activeList = null;
    blocks.push({ type: "p", text: line });
  }

  return (
    <div className="rich">
      {blocks.map((b, i) => {
        if (b.type === "heading") {
          const HeadingTag = `h${b.level}`;

          return (
            <HeadingTag key={i}>
              {renderInline(b.text)}
            </HeadingTag>
          );
        }

        if (b.type === "hr") {
          return <hr key={i} className="rich-divider" />;
        }

        if (b.type === "ul" || b.type === "ol") {
          const ListTag = b.type;

          return (
            <ListTag key={i}>
              {b.items.map((item, j) => (
                <li key={j}>{renderInline(item)}</li>
              ))}
            </ListTag>
          );
        }

        if (b.type === "table") {
          return (
            <div className="rich-table-wrap" key={i}>
              <table className="rich-table">
                <thead>
                  <tr>
                    {b.headers.map((cell, j) => (
                      <th
                        key={j}
                        style={{
                          textAlign: b.alignments[j] || "left",
                        }}
                      >
                        {renderInline(cell)}
                      </th>
                    ))}
                  </tr>
                </thead>

                <tbody>
                  {b.rows.map((row, j) => (
                    <tr key={j}>
                      {b.headers.map((header, k) => (
                        <td
                          key={k}
                          style={{
                            textAlign: b.alignments[k] || "left",
                          }}
                        >
                          {renderInline(row[k] || "")}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }

        return (
          <p key={i}>
            {renderInline(b.text)}
          </p>
        );
      })}
    </div>
  );
}

/* --------------------------------------------------------------- messages */

// function ConfidenceBadge({ level, score }) {
//   return (
//     <span
//       className={`badge badge-${level}`}
//       title="Based on the best similarity score of the retrieved passages"
//     >
//       {LEVEL_LABEL[level] || level} confidence &middot; {score.toFixed(2)}
//     </span>
//   );
// }

function ConfidenceBadge({ level, score }) {
  return (
    <span
      className={`badge badge-${level}`}
      title="Based on the highest dense similarity score. This does not establish whether the retrieved passages contain the answer."
    >
      {LEVEL_LABEL[level] || level} &middot; {score.toFixed(2)}
    </span>
  );
}

function AssistantMessage({ msg }) {
  if (msg.kind === "error") {
    return (
      <div className="notice notice-error pre" role="alert">
        {msg.text}
      </div>
    );
  }

  const isRetrieval = msg.kind === "retrieval";

  return (
    <div className="assistant">
      <div className="meta">
        <ConfidenceBadge level={msg.level} score={msg.score} />
        {!isRetrieval && <span className="meta-item">LLM calls: {msg.calls}</span>}
        {isRetrieval && <span className="meta-item">Retrieval only (no LLM)</span>}
      </div>

      {isRetrieval ? (
        <>
          {msg.level !== "high" && msg.message && <div className="notice notice-info">{msg.message}</div>}
          {msg.chunks.length === 0 && <p className="muted">No passages found.</p>}
          {msg.chunks.map((c, i) => (
            <details className="card" key={i}>
              <summary>
                <span className="card-title">
                  {i + 1}. {c.doc}, p.{c.page}
                </span>
                <span className="chip">similarity {c.dense.toFixed(2)}</span>
              </summary>
              <p className="pre card-body">{c.text}</p>
            </details>
          ))}
        </>
      ) : (
        <>
          {msg.standalone !== msg.original && (
            <p className="muted small">Understood as: {msg.standalone}</p>
          )}
          {msg.level === "medium" && msg.message && (
            <div className="notice notice-warn">{msg.message}</div>
          )}
          <RichText text={msg.text} />

          {msg.sources.length > 0 && (
            <section className="block">
              <h4>Sources ({msg.sources.length})</h4>

              {msg.sources.map((s) => (
                <details className="source" key={s.n}>
                  <summary>
                    <span className="source-n">{s.n}</span>

                    <span>
                      <span className="source-title">
                        {s.doc}, p.{s.page}
                      </span>

                      <span className="muted small">
                        &ldquo;{s.snippet}&hellip;&rdquo;
                      </span>
                    </span>

                    <span className="muted small">View passage</span>
                  </summary>

                  <div className="pre card-body">
                    {s.full_text || s.snippet}
                  </div>
                </details>
              ))}
            </section>
          )}

          {msg.trace.length > 0 && (
            <details className="block">
              <summary>How the agent searched</summary>
              <ol className="trace">
                {msg.trace.map((t, i) => (
                  <li key={i}>
                    <code>{t.query}</code> &rarr; +{t.new_chunks} new passages &rarr; <strong>{t.decision}</strong>
                  </li>
                ))}
              </ol>
            </details>
          )}
        </>
      )}
    </div>
  );
}

function Message({ msg }) {
  if (msg.role === "user") {
    return (
      <div className="row row-user">
        <div className="bubble bubble-user pre">{msg.text}</div>
      </div>
    );
  }

  return (
    <div className="row">
      <div className="avatar" aria-hidden="true">AI</div>
      <div className="bubble bubble-ai">
        <AssistantMessage msg={msg} />
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------- app */

function App() {
  const [status, setStatus] = useState({ documents: [], llm: { connected: false, name: null }, default_model: "" });
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState("agentic");
  const [selected, setSelected] = useState([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const endRef = useRef(null);
  const fileRef = useRef(null);

  useEffect(() => {
    api("/api/status")
      .then((s) => {
        setStatus(s);
        setModel((m) => m || s.default_model);
      })
      .catch((e) => setNotice({ type: "error", text: `Cannot reach the server: ${e.message}` }));
  }, []);

  useEffect(() => {
    const el = endRef.current;
    if (el && el.scrollIntoView) el.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy]);

  const send = useCallback(
    async (text) => {
      const question = text.trim();
      if (!question || busy) return;
      setMessages((m) => [...m, { role: "user", text: question }]);
      setInput("");
      setBusy(true);

      try {
        const data = await api("/api/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, mode, docs: selected.length ? selected : null }),
        });

        setMessages((m) => [...m, { role: "assistant", ...data }]);
      } catch (e) {
        setMessages((m) => [
          ...m,
          { role: "assistant", kind: "error", text: `Could not reach the server: ${e.message}` },
        ]);
      } finally {
        setBusy(false);
      }
    },
    [busy, mode, selected]
  );

  async function uploadFiles(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length) return;

    const form = new FormData();
    files.forEach((f) => form.append("files", f));

    setUploading(true);
    setNotice(null);

    try {
      const data = await api("/api/upload", { method: "POST", body: form });
      setStatus(data);

      const problems = data.results.filter((r) => r.note);

      setNotice(
        problems.length
          ? { type: "warn", text: problems.map((r) => `${r.name}: ${r.note}`).join(" ") }
          : { type: "ok", text: `Loaded ${data.results.map((r) => r.name).join(", ")}.` }
      );
    } catch (e) {
      setNotice({ type: "error", text: `Upload failed: ${e.message}` });
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function loadFolder() {
    setNotice(null);

    try {
      const data = await api("/api/load-folder", { method: "POST" });
      setStatus(data);
      setNotice({ type: "ok", text: "Loaded the PDFs from data/pdfs." });
    } catch (e) {
      setNotice({ type: "error", text: `Could not load the folder: ${e.message}` });
    }
  }

  async function connect() {
    setNotice(null);

    try {
      const data = await api("/api/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: apiKey, model }),
      });

      setStatus(data);
      setNotice({ type: data.ok ? "ok" : "error", text: data.ok ? `Connected: ${data.info}` : data.info });

      if (data.ok) setApiKey("");
    } catch (e) {
      setNotice({ type: "error", text: `Could not connect: ${e.message}` });
    }
  }

  async function clearChat() {
    try {
      await api("/api/reset", { method: "POST" });
    } catch (e) {
      /* ignore: the page is cleared either way */
    }

    setMessages([]);
  }

  function toggleDoc(name) {
    setSelected((cur) => (cur.includes(name) ? cur.filter((d) => d !== name) : [...cur, name]));
  }

  const noDocs = status.documents.length === 0;

  return (
    <div className="app">
      <aside className="sidebar">
        <h1 className="brand">Document QA</h1>

        <section>
          <h2>Documents</h2>
          <div
            className={`drop ${dragging ? "drop-active" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              uploadFiles(e.dataTransfer.files);
            }}
          >
            <p>{uploading ? "Reading PDFs..." : "Drop PDFs here or"}</p>
            <label className="btn btn-secondary">
              Choose files
              <input
                ref={fileRef}
                type="file"
                accept="application/pdf,.pdf"
                multiple
                hidden
                onChange={(e) => uploadFiles(e.target.files)}
              />
            </label>
          </div>

          <button className="btn btn-ghost" onClick={loadFolder}>
            Load PDFs from data/pdfs
          </button>

          {noDocs ? (
            <p className="muted small">No documents loaded yet.</p>
          ) : (
            <ul className="docs">
              {status.documents.map((d) => (
                <li key={d.name}>
                  <label title="Tick to restrict answers to the ticked documents">
                    <input
                      type="checkbox"
                      checked={selected.includes(d.name)}
                      onChange={() => toggleDoc(d.name)}
                    />
                    <span className="doc-name">{d.name}</span>
                    <span className="chip">{d.passages}</span>
                  </label>
                </li>
              ))}
            </ul>
          )}

          {!noDocs && (
            <p className="muted small">
              {selected.length ? `Answers use only the ${selected.length} ticked document(s).` : "Answers use all documents."}
            </p>
          )}
        </section>

        <section>
          <h2>Mode</h2>
          {MODES.map((m) => (
            <label className="radio" key={m.id}>
              <input type="radio" name="mode" checked={mode === m.id} onChange={() => setMode(m.id)} />
              <span>
                <strong>{m.label}</strong>
                <span className="muted small"> {m.hint}</span>
              </span>
            </label>
          ))}
        </section>

        <section>
          <h2>Gemini</h2>
          <p className="status">
            <span className={`dot ${status.llm.connected ? "dot-on" : ""}`} />
            {status.llm.connected ? `Connected: ${status.llm.name}` : "Not connected"}
          </p>

          <label className="field">
            API key
            <input
              type="password"
              autoComplete="off"
              placeholder="or set GEMINI_API_KEY before starting"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
          </label>

          <label className="field">
            Model
            <input type="text" value={model} onChange={(e) => setModel(e.target.value)} />
          </label>

          <button className="btn btn-secondary" onClick={connect}>
            Connect Gemini
          </button>

          <p className="muted small">The key is sent only to this local server and is not saved to disk.</p>
        </section>

        <section>
          <h2>Conversation</h2>
          <button className="btn btn-ghost" onClick={clearChat}>
            Clear conversation
          </button>

          <h3>Examples</h3>
          <div className="examples">
            {EXAMPLES.map((ex) => (
              <button key={ex} className="chip chip-button" onClick={() => send(ex)} disabled={busy}>
                {ex}
              </button>
            ))}
          </div>
        </section>
      </aside>

      <main className="main">
        {notice && (
          <div className={`notice notice-${notice.type} pre`} role="status">
            {notice.text}
            <button className="close" aria-label="Dismiss" onClick={() => setNotice(null)}>
              &times;
            </button>
          </div>
        )}

        <div className="messages" role="log" aria-live="polite">
          {messages.length === 0 && (
            <div className="empty">
              <h2>Ask a question about your documents</h2>
              {/* <p className="muted">
                Answers show their sources (file and page) and a confidence level. Follow-up questions such as
                &ldquo;what are its limitations?&rdquo; are understood from the conversation.
              </p> */}
              <p className="muted">
                Answers show their sources (file and page), along with the retrieval similarity of the
                passages found. Retrieval similarity indicates how closely passages match your question;
                it does not guarantee that they contain the answer. Follow-up questions such as
                &ldquo;what are its limitations?&rdquo; are understood from the conversation.
              </p>
              {noDocs && <div className="notice notice-info">Upload one or more PDFs in the sidebar to begin.</div>}
            </div>
          )}

          {messages.map((m, i) => (
            <Message key={i} msg={m} />
          ))}

          {busy && (
            <div className="row">
              <div className="avatar" aria-hidden="true">AI</div>
              <div className="bubble bubble-ai">
                <span className="typing" aria-label="Working">
                  <i /> <i /> <i />
                </span>
                {mode !== "retrieval" && (
                  <p className="muted small">Free Gemini limits can make this take a minute or two.</p>
                )}
              </div>
            </div>
          )}

          <div ref={endRef} />
        </div>

        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <textarea
            rows={2}
            placeholder="Ask a question about your documents (Enter to send, Shift+Enter for a new line)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(input);
              }
            }}
          />
          <button className="btn btn-primary" type="submit" disabled={busy || !input.trim()}>
            Send
          </button>
        </form>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
