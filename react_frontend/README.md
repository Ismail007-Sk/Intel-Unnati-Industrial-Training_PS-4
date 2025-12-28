# Frontend React Application

This repository contains the **React-based frontend** for the project.  
The frontend is designed to be backend-agnostic and can be integrated with any compatible API implementation.

---

## Project Structure

root/
├── frontend_react/
│ ├── src/
│ ├── public/
│ ├── package.json
│ └── ...


All frontend-related setup and execution must be done **inside the `frontend_react` directory**.

---

## Getting Started

### Prerequisites

Ensure you have the following installed on your system:

- **Node.js** (v18+ recommended)
- **npm** (comes bundled with Node.js)

You can verify installation using:
```bash
node -v
npm -v
```

## Installation & Running the App

Step 1: Navigate to the Frontend Directory
cd frontend_react

Step 2: Install Dependencies

Install all required dependencies defined in package.json:

npm install

Step 3: Run the Development Server

Start the application in development mode:

npm run dev
After successful startup, the terminal will display a local development URL
(commonly http://localhost:5173 or http://localhost:3000).

Open the URL in your browser to access the application.

Step 4: Production Build (Optional)
To generate an optimized production build:

bash
Copy code
npm run build
The compiled output will be available in the dist/ directory.

### Backend Integration

This frontend does not ship with a preconfigured backend.

You are expected to wire the backend endpoints yourself based on your API design.

### API Configuration

Backend configuration should be handled using environment variables.

Create a .env file inside the frontend_react directory:
