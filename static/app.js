// PageStream: a streaming-style frontend for the Book Recommendation API.
// It only talks to the three endpoints in book_api.py.
// All text is inserted with textContent (never innerHTML), so book data can't inject HTML.

const $ = (id) => document.getElementById(id);
const smooth = matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";

// ---------- API calls ----------
async function request(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      /* response was not JSON */
    }
    throw new Error(message);
  }
  return res.json();
}

const api = {
  books: () => request("/api/books"),
  search: (query, topK = 6) =>
    request(`/api/books/search?top_k=${topK}`, {
      method: "POST",
      headers: { "Content-Type": "text/plain" },
      body: query,
    }),
  similar: (title, topK = 5) =>
    request(`/api/books/similar?title=${encodeURIComponent(title)}&top_k=${topK}`),
};

// ---------- Building blocks ----------
function hueFor(title) {
  let h = 0;
  for (const ch of title) h = (h * 31 + ch.codePointAt(0)) % 360;
  return h;
}

function makeCover(title) {
  const cover = document.createElement("div");
  cover.className = "cover";
  cover.style.setProperty("--h", hueFor(title));
  cover.dataset.initial = title.charAt(0);
  const label = document.createElement("span");
  label.className = "cover-title";
  label.textContent = title;
  cover.append(label);
  return cover;
}

function makeCard(book) {
  const card = document.createElement("button");
  card.type = "button";
  card.className = "card";
  card.append(makeCover(book.title));

  const meta = document.createElement("div");
  meta.className = "card-meta";
  const name = document.createElement("span");
  name.className = "card-title";
  name.textContent = book.title;
  meta.append(name);

  if (typeof book.score === "number") {
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = `${book.score.toFixed(2)} match`;
    badge.title = "Cosine similarity (1.00 = identical meaning)";
    meta.append(badge);
  }

  card.append(meta);
  card.addEventListener("click", () => openDetails(book));
  return card;
}

function fillTrack(track, books) {
  track.replaceChildren(...books.map(makeCard));
}

function showSkeletons(track, count = 5) {
  const items = Array.from({ length: count }, () => {
    const s = document.createElement("div");
    s.className = "skeleton";
    return s;
  });
  track.replaceChildren(...items);
}

function setStatus(message, isError = false) {
  const status = $("status");
  status.textContent = message;
  status.classList.toggle("error", isError);
}

// Each section remembers its latest request, so a slow old answer
// can't overwrite a newer one if the user clicks quickly.
const latest = { search: 0, similar: 0, details: 0 };

// ---------- Hero (featured book) ----------
function renderHero(book) {
  const text = document.createElement("div");

  const eyebrow = document.createElement("p");
  eyebrow.className = "eyebrow";
  eyebrow.textContent = "Featured pick";

  const title = document.createElement("h1");
  title.textContent = book.title;

  const desc = document.createElement("p");
  desc.textContent = book.description;

  const actions = document.createElement("div");
  actions.className = "actions";

  const more = document.createElement("button");
  more.type = "button";
  more.className = "btn btn-primary";
  more.textContent = "More like this";
  more.addEventListener("click", () => showSimilarRow(book));

  const details = document.createElement("button");
  details.type = "button";
  details.className = "btn btn-secondary";
  details.textContent = "Details";
  details.addEventListener("click", () => openDetails(book));

  actions.append(more, details);
  text.append(eyebrow, title, desc, actions);
  $("hero").replaceChildren(makeCover(book.title), text);
}

// ---------- Feature 2: "Because you liked X" row ----------
async function showSimilarRow(book) {
  const token = ++latest.similar;
  const row = $("similar-row");
  row.hidden = false;
  $("similar-title").textContent = `Because you liked ${book.title}`;
  showSkeletons($("similar-track"));
  row.scrollIntoView({ behavior: smooth, block: "nearest" });

  try {
    const results = await api.similar(book.title);
    if (token !== latest.similar) return;
    fillTrack($("similar-track"), results);
  } catch (err) {
    if (token !== latest.similar) return;
    $("similar-track").replaceChildren();
    setStatus(err.message, true);
  }
}

// ---------- Details pop-up ----------
async function openDetails(book) {
  const token = ++latest.details;
  const dialog = $("details");
  $("details-cover").replaceChildren(makeCover(book.title));
  $("details-title").textContent = book.title;
  $("details-desc").textContent = book.description ?? "";
  showSkeletons($("details-similar"), 4);
  if (!dialog.open) dialog.showModal();

  try {
    const results = await api.similar(book.title, 4);
    if (token !== latest.details) return;
    fillTrack($("details-similar"), results);
  } catch (err) {
    if (token !== latest.details) return;
    const p = document.createElement("p");
    p.textContent = err.message;
    $("details-similar").replaceChildren(p);
  }
}

$("details-close").addEventListener("click", () => $("details").close());
$("details").addEventListener("click", (e) => {
  if (e.target === $("details")) $("details").close(); // click on the dark backdrop
});

// ---------- Feature 1: semantic search ----------
$("search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = $("search-input").value.trim();
  if (!query) return;

  const token = ++latest.search;
  const row = $("search-row");
  row.hidden = false;
  $("search-title").textContent = `Results for “${query}”`;
  showSkeletons($("search-track"));
  setStatus("");
  row.scrollIntoView({ behavior: smooth, block: "nearest" });

  try {
    const results = await api.search(query);
    if (token !== latest.search) return;
    fillTrack($("search-track"), results);
  } catch (err) {
    if (token !== latest.search) return;
    $("search-track").replaceChildren();
    setStatus(`Search failed: ${err.message}`, true);
  }
});

// ---------- Start ----------
async function init() {
  try {
    const books = await api.books();
    if (!books.length) {
      setStatus("No books yet. Add some to resources/books.json.");
      return;
    }
    fillTrack($("all-track"), books);
    renderHero(books[Math.floor(Math.random() * books.length)]);
  } catch (err) {
    setStatus(`Could not load books: ${err.message}. Is the server running?`, true);
  }
}

init();
