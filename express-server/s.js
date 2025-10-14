// server.js
const express = require("express");
const jwt = require("jsonwebtoken");
const app = express();
const PORT = 4000;

app.use(express.json());

// Mock Data
let books = [
  { id: 1, title: "Atomic Habits", author: "James Clear", genre: "Self-help", year: 2018 },
  { id: 2, title: "Sapiens", author: "Yuval Noah Harari", genre: "History", year: 2011 },
  { id: 3, title: "The Pragmatic Programmer", author: "Andy Hunt", genre: "Programming", year: 1999 },
  { id: 4, title: "Clean Code", author: "Robert C. Martin", genre: "Programming", year: 2008 },
];

// Secret for JWT
const SECRET_KEY = "supersecret";

// Middleware to check authentication
function authMiddleware(req, res, next) {
  const token = req.headers["authorization"];
  if (!token) return res.status(403).json({ message: "Token required" });
  try {
    jwt.verify(token.split(" ")[1], SECRET_KEY);
    next();
  } catch (err) {
    return res.status(401).json({ message: "Invalid token" });
  }
}

/* 
   1. LOGIN - Returns JWT token 
   Request: { "username": "admin", "password": "1234" }
*/
app.post("/api/login", (req, res) => {
  const { username, password } = req.body;
  if (username === "admin" && password === "1234") {
    const token = jwt.sign({ user: username }, SECRET_KEY, { expiresIn: "1h" });
    return res.json({ token });
  }
  res.status(401).json({ message: "Invalid credentials" });
});

/* 
   2. GET Books with Filtering + Pagination
   Example: /api/books?genre=Programming&page=1&limit=2
*/
app.get("/api/books", (req, res) => {
  let { genre, page = 1, limit = 2 } = req.query;
  let filtered = books;

  if (genre) {
    filtered = filtered.filter((b) => b.genre.toLowerCase() === genre.toLowerCase());
  }

  // Pagination
  const start = (page - 1) * limit;
  const end = start + parseInt(limit);
  const paginated = filtered.slice(start, end);

  res.json({
    total: filtered.length,
    page: parseInt(page),
    limit: parseInt(limit),
    data: paginated,
  });
});

/* 
   3. GET Book details with "Related books" (nested response)
   Example: /api/books/1
*/
app.get("/api/books/:id", (req, res) => {
  const book = books.find((b) => b.id === parseInt(req.params.id));
  if (!book) return res.status(404).json({ message: "Book not found" });

  const related = books.filter((b) => b.genre === book.genre && b.id !== book.id);
  res.json({ ...book, relatedBooks: related });
});

/* 
   4. POST Add Book (Auth Protected)
   Headers: Authorization: Bearer <token>
*/
app.post("/api/books", authMiddleware, (req, res) => {
  const { title, author, genre, year } = req.body;
  const newBook = { id: books.length + 1, title, author, genre, year };
  books.push(newBook);
  res.status(201).json(newBook);
});

/* 
   5. GET Async External Simulation (simulate long processing)
   Example: /api/recommendations
*/
app.get("/api/recommendations", async (req, res) => {
  // Fake delay
  await new Promise((resolve) => setTimeout(resolve, 2000));

  const recommendations = books.sort(() => 0.5 - Math.random()).slice(0, 2);
  res.json({ recommendations });
});

// Start Server
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});
