# Simple Docker Project

This project demonstrates how to set up a simple Node.js application with Docker.

## Project Structure

```
simple-docker-project
├── Dockerfile
├── .dockerignore
├── src
│   └── app.js
├── package.json
└── README.md
```

## Getting Started

To build and run the Docker container for this project, follow these steps:

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd simple-docker-project
   ```

2. **Build the Docker image:**
   ```
   docker build -t simple-docker-project .
   ```

3. **Run the Docker container:**
   ```
   docker run -p 3000:3000 simple-docker-project
   ```

4. **Access the application:**
   Open your browser and go to `http://localhost:3000`.

## Prerequisites

- Docker installed on your machine.
- Basic knowledge of Node.js and Docker.

## License

This project is licensed under the MIT License.