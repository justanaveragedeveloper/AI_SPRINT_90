// test_bridge_advanced.js
// Run with: node test_bridge_advanced.js

const BASE_URL = "http://127.0.0.1:8000";
// Matching the secure validation fallback keys from day6_api.py
const DEV_API_KEY = "dev-secret-key-12345";
const ADMIN_API_KEY = "admin-secret-key-67890";

async function testAutoChunking() {
  console.log(
    "🚀 Testing Auto-Chunking Integration with Hardened Auth Bridge...\n",
  );

  try {
    // 1. Test small document (no chunking)
    console.log("📝 TEST 1: Small document (no chunking)");
    const smallDoc = {
      id: "small_doc_001",
      text: "This is a short document that won't trigger chunking.",
      metadata: { type: "short", category: "test" },
    };

    let response = await fetch(`${BASE_URL}/upsert`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": DEV_API_KEY,
      },
      body: JSON.stringify(smallDoc),
    });
    let data = await response.json();
    console.log("Response:", JSON.stringify(data, null, 2));
    console.log("✅ Small document indexed successfully\n");

    // 2. Test large resume (auto-chunking)
    console.log("📝 TEST 2: Large resume (auto-chunking expected)");
    const largeResume = {
      id: "candidate_alex_chen",
      text: `
ALEXANDER CHEN
Senior Full Stack Developer | Cloud Architect
Email: alex.chen@techinnovations.com | Phone: (555) 123-4567

PROFESSIONAL SUMMARY
Innovative Full Stack Developer with 8+ years of experience in building scalable web applications and cloud-native solutions. Expert in React, Node.js, and AWS infrastructure. Proven track record of leading development teams and delivering high-impact projects from concept to deployment. Strong advocate for clean code, automated testing, and DevOps best practices.

TECHNICAL SKILLS
Frontend: React 18+, Vue.js, Angular, TypeScript, Redux, Next.js, Tailwind CSS, Material-UI
Backend: Node.js, Express, Python (Django/Flask), Java Spring Boot, GraphQL, REST APIs
Database: PostgreSQL, MongoDB, Redis, Elasticsearch, DynamoDB
Cloud & DevOps: AWS (EC2, S3, Lambda, API Gateway, CloudFormation), Docker, Kubernetes, Jenkins, GitHub Actions
Testing: Jest, PyTest, Mocha, Selenium, Cypress

WORK EXPERIENCE
Senior Full Stack Developer | TechInnovations Inc. | San Francisco, CA
January 2021 - Present
- Architected and deployed a microservices-based e-commerce platform handling 500K+ daily active users, achieving 99.99% uptime using AWS ECS and auto-scaling groups
- Migrated monolithic application to microservices architecture, reducing deployment time by 65% and improving fault isolation
- Implemented real-time analytics dashboard using WebSocket connections and Redis pub/sub, processing 10K+ events per second
- Led migration of frontend from AngularJS to React 18, improving performance scores from 45 to 92 on Lighthouse
- Mentored 5 junior developers, conducting code reviews and leading weekly knowledge-sharing sessions
- Reduced cloud costs by 35% through implementing AWS Lambda functions and optimizing EC2 instance usage

Full Stack Developer | CloudScale Solutions | Austin, TX
June 2018 - December 2020
- Developed RESTful APIs using Node.js and Express serving 1M+ requests monthly with average response time under 200ms
- Built responsive frontend components with React and Redux, increasing user engagement by 40%
- Implemented JWT-based authentication and role-based access control for multi-tenant SaaS product
- Optimized MongoDB queries and indexing, reducing database response time from 800ms to 50ms
- Created automated testing suite using Jest and Supertest, achieving 85% code coverage
- Deployed and managed applications on AWS EC2 with load balancers and auto-scaling groups
`,
      metadata: {
        role: "Engineering",
        experience_years: 8,
        file_type: "resume",
        contact_info: {
          email: "alex.chen@techinnovations.com",
          phone: "555-123-4567",
        },
      },
    };

    response = await fetch(`${BASE_URL}/upsert`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": DEV_API_KEY,
      },
      body: JSON.stringify(largeResume),
    });
    data = await response.json();
    console.log("Chunking Response:", JSON.stringify(data, null, 2));
    console.log(
      `✅ Large document split into ${data.total_chunks_indexed} chunks\n`,
    );

    // 3. Search across chunks
    console.log(
      "🔍 TEST 3: Semantic search across chunked document (User Key - Filtered Output)",
    );
    const searchQuery = {
      query: "cloud cost optimization and AWS experience",
      n_results: 2,
    };

    response = await fetch(`${BASE_URL}/search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": DEV_API_KEY,
      },
      body: JSON.stringify(searchQuery),
    });
    data = await response.json();
    console.log("Search Results:", JSON.stringify(data, null, 2));

    // 4. Get admin metrics
    console.log("\n📊 TEST 4: System metrics (Admin Key)");
    response = await fetch(`${BASE_URL}/metrics`, {
      headers: { "X-API-Key": ADMIN_API_KEY },
    });
    data = await response.json();
    console.log("Metrics:", JSON.stringify(data, null, 2));
  } catch (error) {
    console.error("❌ Test failed:", error.message);
  }
}

testAutoChunking();
