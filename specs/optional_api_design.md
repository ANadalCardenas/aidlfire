# Catalonia Wildfire Detection System

## Optional: API Design (Simplified for Capstone)

**Project Type:** Postgraduate Capstone Project
**Domain:** Computer Vision / Remote Sensing
**Region:** Catalonia, Spain
**Date:** January 2026

---

**Note:** API implementation is **optional** for this capstone project. If using Streamlit for the UI, you can load the model directly in Streamlit without needing a separate API layer. This document provides API design guidance if you choose to implement an API for separation of concerns or to demonstrate API design skills.

API design defines how external systems (web UI, mobile apps, other services) interact with the wildfire detection system. For a capstone project, the API should be simple but demonstrate core functionality.

**Design Philosophy:**

- **Simplified for capstone:** Focus on demonstrating model inference, not production-ready features
- **RESTful principles:** Use standard HTTP methods and status codes
- **Clear contracts:** Well-defined request/response schemas
- **Documentation:** Automatic API documentation for easy testing

---

### 1. Technology Choice

**Framework: FastAPI**

**Why FastAPI?**

| Framework | Pros | Cons | Verdict |
|-----------|------|------|---------|
| **FastAPI** | ✅ Async support<br>✅ Auto OpenAPI docs<br>✅ Type validation<br>✅ High performance<br>✅ Easy to learn | ⚠️ Python-only | **⭐ Recommended** |
| Flask | ✅ Simple<br>✅ Widely used | ⚠️ No async (slower for I/O)<br>⚠️ Manual validation<br>⚠️ Manual docs | Good for simple APIs |
| Django REST | ✅ Full framework<br>✅ Admin panel | ⚠️ Overkill for simple API<br>⚠️ More complex | Too heavy for capstone |

**Rationale for FastAPI:**

**Async support:**
- **Why it matters:** Fetching satellite imagery is I/O-bound (waiting for external APIs)
- **Performance:** Async allows handling multiple requests efficiently
- **Real-world relevance:** Demonstrates modern API design patterns

**Automatic OpenAPI documentation:**
- **What it is:** Interactive API documentation (Swagger UI) generated automatically
- **Why it matters:** Easy testing, clear API contract, professional appearance
- **Capstone value:** Demonstrates API design best practices

**Type hints and Pydantic validation:**
- **What it is:** Request/response validation using Python type hints
- **Why it matters:** Catches errors early, clear API contracts, better developer experience
- **Academic value:** Shows understanding of modern Python practices

**High performance:**
- **Why it matters:** Fast response times improve user experience
- **Benchmark:** FastAPI is one of the fastest Python frameworks
- **Capstone note:** Performance is less critical for demo, but good practice

**Simple to learn:**
- **Why it matters:** Reduces learning curve, more time for model work
- **Documentation:** Excellent documentation, many examples
- **Community:** Large community, easy to find help

---

### 2. Simplified Endpoint Summary

**Why Simplified?**

For capstone, the API should demonstrate core functionality without production complexity. Focus is on model inference, not enterprise features.

**Simplified Features Rationale:**

**Single synchronous endpoint:**
- **Why:** Simplest to implement and understand
- **Alternative:** Async jobs for long-running tasks (more complex, not needed for capstone)
- **Trade-off:** Synchronous means users wait for response (acceptable for demo)

**Basic health check:**
- **Why:** Essential for deployment monitoring, simple to implement
- **What it checks:** API is running, model is loaded, dependencies available
- **Use case:** Deployment platforms (GCP Cloud Run) use health checks

**Optional authentication:**
- **Why optional:** Simplifies demo, no user management needed
- **If implemented:** Simple API key in header (no database, no user accounts)
- **Production note:** Real systems need proper authentication, but not for capstone

**Removed complex features:**
- **Alert subscriptions:** Requires database, user management - too complex for capstone
- **Spread analysis:** Can be done client-side or omitted
- **History tracking:** Requires database - can be simplified or omitted

#### 2.1 Core Detection Endpoints

**POST `/api/v1/detect`**

**What it does:** Detects fires in satellite imagery for a given bounding box and date.

**Why POST:**
- **Request body:** Bounding box and date are complex parameters (not simple query params)
- **Semantic correctness:** POST is appropriate for operations that process data
- **Future extensibility:** Easy to add more parameters (threshold, bands, etc.)

**Why `/api/v1/`:**
- **Versioning:** Allows future API versions without breaking changes
- **Best practice:** Standard REST API versioning pattern
- **Capstone note:** v1 is sufficient, but shows understanding of API design

**GET `/api/v1/health`**

**What it does:** Returns API health status.

**Why GET:**
- **Idempotent:** Health check doesn't change state
- **Cacheable:** Health status can be cached (though typically not)
- **Standard:** Health checks are typically GET endpoints

**Response fields:**
- **Status:** "healthy" or "unhealthy"
- **Model loaded:** Whether PyTorch model is loaded
- **Dependencies:** Status of external services (Sentinel API, etc.)
- **Timestamp:** When health check was performed

---

### 3. Request/Response Schemas

**Why Defined Schemas?**

Clear request/response schemas ensure API contracts are well-defined, enable automatic validation, and make API easier to use and test.

#### Detection Request

**Bounding Box (bbox):**

**Why bounding box:**
- **Flexible region selection:** Users can analyze any geographic region
- **Standard format:** Bounding boxes are standard in GIS and remote sensing
- **Simple to specify:** Four coordinates (min_lon, min_lat, max_lon, max_lat)

**Coordinate system:** WGS84 (EPSG:4326) - standard web coordinate system

**Validation:**
- **Range checks:** Longitude [-180, 180], Latitude [-90, 90]
- **Order checks:** min_lon < max_lon, min_lat < max_lat
- **Size limits:** Maximum bounding box size (e.g., 1° × 1°) to prevent excessive processing

**Date:**

**Why ISO format (YYYY-MM-DD):**
- **Standard:** ISO 8601 is international standard for dates
- **Unambiguous:** No confusion about date format (MM/DD vs DD/MM)
- **Sortable:** ISO dates sort correctly as strings
- **Validation:** Easy to validate format and range

**Date constraints:**
- **Past dates only:** Can't detect fires in future (satellite imagery is historical)
- **Minimum date:** Sentinel-2 data available from 2015
- **Maximum date:** Typically up to 2-3 days ago (processing delay)

**Threshold:**

**Why optional:**
- **Default value:** 0.5 is reasonable default (balanced precision/recall)
- **User control:** Advanced users may want to adjust threshold
- **Flexibility:** Allows experimentation without code changes

**Range:** 0.0 to 1.0 (probability range)

#### Detection Response

**Fire detected (boolean):**

**Why boolean:**
- **Quick check:** Users can quickly determine if any fire was found
- **Simple logic:** Enables conditional behavior (alert if true, skip if false)
- **Clear semantics:** Binary outcome is easy to understand

**Total area (hectares):**

**Why hectares:**
- **Standard unit:** Fire area is typically reported in hectares
- **Meaningful scale:** Appropriate for fire sizes
- **User familiarity:** Emergency services use hectares

**Fire areas (GeoJSON array):**

**Why GeoJSON:**
- **Standard format:** Widely supported by web maps and GIS software
- **Self-contained:** Includes coordinates and metadata in single format
- **Web-friendly:** JSON format works well with web applications

**Why array:**
- **Multiple fires:** Single image may contain multiple fire events
- **Individual polygons:** Each fire gets its own polygon for detailed analysis
- **Flexibility:** Can handle any number of fires

**Processing time (seconds):**

**Why include:**
- **Performance monitoring:** Helps identify slow requests
- **User feedback:** Users know how long operation took
- **Debugging:** Helps identify performance bottlenecks
- **Academic value:** Demonstrates understanding of performance metrics

---

### 4. Authentication (Optional)

**Why Optional for Capstone?**

Authentication adds complexity (user management, tokens, security) that isn't essential for demonstrating model inference. For capstone demo, authentication can be skipped.

**If Implemented: Simple API Key**

**Why API key (not OAuth/JWT):**
- **Simplicity:** API key is simplest authentication method
- **Sufficient for demo:** Provides basic access control
- **No user management:** No need for user accounts, passwords, registration
- **Easy to implement:** Single header check, no complex token validation

**Implementation:**
- **Header:** `X-API-Key: your-api-key-here`
- **Storage:** Environment variable or simple config file (not database)
- **Validation:** Check API key matches configured key(s)

**Production note:** Real systems need proper authentication (OAuth, JWT, user management), but API key is sufficient for capstone demo.

---

### 5. Error Codes

**Why Standard HTTP Codes?**

Using standard HTTP status codes ensures API follows REST conventions, is easy to understand, and works well with HTTP clients and tools.

**200 Success:**
- **When:** Request processed successfully, fires detected or not detected
- **Why:** Standard success code for GET/POST requests
- **Response:** Always includes detection results (even if no fires found)

**400 Bad Request:**
- **When:** Invalid parameters (invalid bbox, invalid date format, out of range values)
- **Why:** Client error - user provided invalid input
- **Response:** Error message explaining what was wrong

**404 Not Found:**
- **When:** No satellite imagery available for requested date/region
- **Why:** Resource not found - imagery doesn't exist
- **Alternative:** Could use 400 (bad request), but 404 is more semantically correct

**500 Internal Server Error:**
- **When:** Server-side error (model loading failed, processing error, bug)
- **Why:** Server error - not user's fault
- **Response:** Generic error message (don't expose internal details)

**503 Service Unavailable:**
- **When:** External service down (Sentinel API unavailable, model not loaded)
- **Why:** Service temporarily unavailable - different from 500 (permanent error)
- **Response:** Error message indicating service is temporarily unavailable

**Error Response Format:**
- **Consistent structure:** All errors use same JSON format
- **Error code:** HTTP status code
- **Error message:** Human-readable error description
- **Error details:** Optional additional information (for 400 errors, list invalid fields)

---

### 6. Implementation Checklist

**If Implementing API:**

- [ ] Set up FastAPI project structure
- [ ] Implement model loading function (load PyTorch .pt model)
- [ ] Implement inference function (preprocessing, model forward pass, postprocessing)
- [ ] Implement Sentinel-2 imagery fetching (Copernicus Data Space API)
- [ ] Implement `POST /api/v1/detect` endpoint
- [ ] Implement `GET /api/v1/health` endpoint
- [ ] Add request validation (Pydantic models)
- [ ] Implement error handling (400, 404, 500, 503)
- [ ] Test endpoints with sample requests
- [ ] Generate OpenAPI documentation (automatic with FastAPI)
- [ ] Deploy to GCP Cloud Run (or alternative)
- [ ] Test deployed API

**Time Estimate:** 2 weeks (Phase 3 in original timeline)

**Note:** If using Streamlit with direct model loading, API implementation can be skipped entirely to save time and focus on model work.
