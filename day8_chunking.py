import uuid


def split_text_into_chunks(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list:
    """
    Slices a continuous text block into logical semantic segments
    safely bound by character constraints with an overlap window.
    Breaks cleanly on word boundaries to preserve semantic integrity.

    Returns a list of dictionaries, each containing:
    - id: Unique identifier for the chunk
    - text: The chunk content
    - metadata: Index position and length info
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        # Calculate target end position
        end = start + chunk_size

        # If we can capture the rest of the text, take it all
        if end >= text_len:
            end = text_len
            chunk_text = text[start:end]

            # Create chunk dictionary with UUID
            chunks.append(
                {
                    "id": str(uuid.uuid4()),  # Now actually used!
                    "text": chunk_text,
                    "metadata": {
                        "start_index": start,
                        "end_index": end,
                        "chunk_size": len(chunk_text),
                        "is_final_chunk": True,
                    },
                }
            )
            break

        # SMART ADAPTATION: Look backward for a clean break (space or newline)
        original_end = end
        while end > start and text[end - 1] not in [" ", "\n"]:
            end -= 1
            if original_end - end > 100:  # Safety guardrail
                end = original_end
                break

        # Extract the clean chunk
        chunk_text = text[start:end]

        # Create chunk dictionary with UUID
        chunks.append(
            {
                "id": str(uuid.uuid4()),  # Unique identifier for vector DB
                "text": chunk_text,
                "metadata": {
                    "start_index": start,
                    "end_index": end,
                    "chunk_size": len(chunk_text),
                    "was_adjusted": end != original_end,
                    "overlap_amount": chunk_overlap if start > 0 else 0,
                },
            }
        )

        # Adjust next start position based on ACTUAL end minus overlap
        start = end - chunk_overlap

        # Safety guardrail to prevent infinite loops
        if start >= end:
            start = end + (chunk_size - chunk_overlap)

    return chunks


# Alternative: Simpler version without UUID if you don't need it
def split_text_into_chunks_simple(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list:
    """
    Same splitting logic but returns just strings (no UUIDs)
    Use this if you don't need unique identifiers.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        if end >= text_len:
            chunks.append(text[start:text_len])
            break

        original_end = end
        while end > start and text[end - 1] not in [" ", "\n"]:
            end -= 1
            if original_end - end > 100:
                end = original_end
                break

        chunks.append(text[start:end])
        start = end - chunk_overlap

        if start >= end:
            start = end + (chunk_size - chunk_overlap)

    return chunks


# Test with the resume
huge_resume = """
ALEXANDER CHEN
Senior Full Stack Developer | Cloud Architect
Email: alex.chen@techinnovations.com | Phone: (555) 123-4567 | GitHub: github.com/alexchen

PROFESSIONAL SUMMARY
Innovative Full Stack Developer with 8+ years of experience in building scalable web applications and cloud-native solutions. Expert in React, Node.js, and AWS infrastructure. Proven track record of leading development teams and delivering high-impact projects from concept to deployment. Strong advocate for clean code, automated testing, and DevOps best practices.

TECHNICAL SKILLS
Frontend: React 18+, Vue.js, Angular, TypeScript, Redux, Next.js, Tailwind CSS, Material-UI
Backend: Node.js, Express, Python (Django/Flask), Java Spring Boot, GraphQL, REST APIs
Database: PostgreSQL, MongoDB, Redis, Elasticsearch, DynamoDB
Cloud & DevOps: AWS (EC2, S3, Lambda, API Gateway, CloudFormation), Docker, Kubernetes, Jenkins, GitHub Actions
Testing: Jest, PyTest, Mocha, Selenium, Cypress
Monitoring: Prometheus, Grafana, ELK Stack, New Relic
Version Control: Git, GitFlow, GitHub, GitLab

WORK EXPERIENCE

Senior Full Stack Developer | TechInnovations Inc. | San Francisco, CA
January 2021 - Present
- Architected and deployed a microservices-based e-commerce platform handling 500K+ daily active users, achieving 99.99% uptime using AWS ECS and auto-scaling groups
- Migrated monolithic application to microservices architecture, reducing deployment time by 65% and improving fault isolation
- Implemented real-time analytics dashboard using WebSocket connections and Redis pub/sub, processing 10K+ events per second
- Led migration of frontend from AngularJS to React 18, improving performance scores from 45 to 92 on Lighthouse
- Mentored 5 junior developers, conducting code reviews and leading weekly knowledge-sharing sessions
- Reduced cloud costs by 35% through implementing AWS Lambda functions and optimizing EC2 instance usage
- Set up CI/CD pipeline using Jenkins and GitHub Actions, reducing manual deployment errors by 90%

Full Stack Developer | CloudScale Solutions | Austin, TX
June 2018 - December 2020
- Developed RESTful APIs using Node.js and Express serving 1M+ requests monthly with average response time under 200ms
- Built responsive frontend components with React and Redux, increasing user engagement by 40%
- Implemented JWT-based authentication and role-based access control for multi-tenant SaaS product
- Optimized MongoDB queries and indexing, reducing database response time from 800ms to 50ms
- Created automated testing suite using Jest and Supertest, achieving 85% code coverage
- Deployed and managed applications on AWS EC2 with load balancers and auto-scaling groups
- Collaborated with product team to define technical requirements and deliver 15+ major features on schedule
"""

# Process with UUID version
print("=" * 80)
print("VERSION 1: WITH UUID (For Vector Database Storage)")
print("=" * 80)

processed_chunks = split_text_into_chunks(huge_resume)

print(f"Original Text Length: {len(huge_resume)} characters.")
print(f"Generated {len(processed_chunks)} distinct chunks for vector indexing.\n")

for idx, chunk_obj in enumerate(processed_chunks):
    print(f"\n--- CHUNK {idx+1} ---")
    print(f"UUID: {chunk_obj['id']}")
    print(f"Length: {chunk_obj['metadata']['chunk_size']} chars")
    print(
        f"Indices: {chunk_obj['metadata']['start_index']} -> {chunk_obj['metadata']['end_index']}"
    )
    print(f"Text preview: {chunk_obj['text'][:150].replace(chr(10), ' ')}...")

# Process with simple version (no UUID)
print("\n" + "=" * 80)
print("VERSION 2: SIMPLE STRINGS (No UUID - Cleaner Code)")
print("=" * 80)

simple_chunks = split_text_into_chunks_simple(huge_resume)
print(f"Generated {len(simple_chunks)} chunks as plain strings")
print(f"Example chunk 1: {simple_chunks[0][:150].replace(chr(10), ' ')}...")
