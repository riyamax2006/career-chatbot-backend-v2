"""
Career Dataset for NLP-Based Recommendation Engine.

Each career includes rich text descriptions optimized for TF-IDF matching.
Salaries are in LPA (Lakhs Per Annum).
Risk levels: low, medium, high.
"""

CAREERS = [
    # --- TECHNOLOGY ---
    {
        "role": "Software Development Engineer (Product)",
        "category": "Technology",
        "description": (
            "Build and maintain software products using programming languages like Python, Java, "
            "JavaScript, C++. Work on web applications, mobile apps, APIs, and backend systems. "
            "Skills include coding, debugging, software architecture, agile development, Git, "
            "cloud computing AWS Azure. High-growth role in product companies and startups."
        ),
        "salaries": {"entry": 10, "mid": 20, "senior": 45},
        "risk": "medium",
        "keywords": ["coding", "programming", "software", "developer", "engineer", "tech", "IT"]
    },
    {
        "role": "Service-Based Software Engineer",
        "category": "Technology",
        "description": (
            "Work in IT services companies like TCS, Infosys, Wipro on client projects. "
            "Maintain enterprise software, provide support, work on legacy systems. "
            "Stable corporate environment with structured career path. "
            "Skills include Java, .NET, SQL, enterprise software, consulting."
        ),
        "salaries": {"entry": 4, "mid": 8, "senior": 15},
        "risk": "low",
        "keywords": ["IT services", "consulting", "enterprise", "corporate", "stable"]
    },
    {
        "role": "Data Scientist",
        "category": "Technology",
        "description": (
            "Analyze large datasets using machine learning, statistical modeling, Python, R, SQL. "
            "Build predictive models, create data visualizations, extract insights from data. "
            "Work with AI, deep learning, neural networks, TensorFlow, scikit-learn. "
            "High demand role requiring statistics, mathematics, and programming skills."
        ),
        "salaries": {"entry": 8, "mid": 18, "senior": 35},
        "risk": "medium",
        "keywords": ["data", "machine learning", "AI", "analytics", "statistics", "python"]
    },
    {
        "role": "Cybersecurity Analyst",
        "category": "Technology",
        "description": (
            "Protect computer systems and networks from cyber threats, hackers, malware. "
            "Conduct security audits, penetration testing, incident response. "
            "Skills include network security, encryption, firewalls, SIEM, ethical hacking. "
            "Critical role with growing demand as digital threats increase."
        ),
        "salaries": {"entry": 6, "mid": 14, "senior": 28},
        "risk": "medium",
        "keywords": ["security", "hacking", "networks", "protection", "cyber"]
    },

    # --- MANAGEMENT & BUSINESS ---
    {
        "role": "Management Consultant",
        "category": "Business",
        "description": (
            "Advise organizations on strategy, operations, and business transformation. "
            "Work with McKinsey, BCG, Bain or similar firms. Analyze business problems, "
            "create presentations, develop recommendations for C-suite executives. "
            "Requires strong analytical skills, MBA preferred, intense travel and long hours."
        ),
        "salaries": {"entry": 14, "mid": 25, "senior": 50},
        "risk": "high",
        "keywords": ["consulting", "strategy", "business", "MBA", "corporate", "management"]
    },
    {
        "role": "Investment Banker",
        "category": "Finance",
        "description": (
            "Work in financial services on mergers and acquisitions, IPOs, capital raising. "
            "Analyze company valuations, create financial models, pitch to clients. "
            "Elite finance role at Goldman Sachs, Morgan Stanley, JP Morgan. "
            "Extremely high compensation but demanding hours and high stress environment."
        ),
        "salaries": {"entry": 15, "mid": 30, "senior": 60},
        "risk": "high",
        "keywords": ["finance", "banking", "investment", "Wall Street", "money", "stocks"]
    },
    {
        "role": "Digital Marketing Specialist",
        "category": "Marketing",
        "description": (
            "Plan and execute online marketing campaigns using SEO, SEM, social media, "
            "content marketing, email marketing, Google Ads, Facebook Ads. "
            "Analyze campaign performance, optimize conversion rates, grow brand awareness. "
            "Creative role combining analytics and communication skills."
        ),
        "salaries": {"entry": 4, "mid": 9, "senior": 18},
        "risk": "medium",
        "keywords": ["marketing", "digital", "social media", "advertising", "SEO", "creative"]
    },
    {
        "role": "Human Resources Generalist",
        "category": "HR",
        "description": (
            "Manage employee relations, recruitment, onboarding, payroll, benefits. "
            "Handle HR policies, training programs, performance management. "
            "Stable back-office corporate function with predictable career progression. "
            "Good for people who enjoy working with employees and organizational development."
        ),
        "salaries": {"entry": 3.5, "mid": 7, "senior": 14},
        "risk": "low",
        "keywords": ["HR", "human resources", "recruitment", "people", "corporate"]
    },
    {
        "role": "Chartered Accountant (Corporate)",
        "category": "Finance",
        "description": (
            "Handle accounting, auditing, taxation, financial reporting for corporations. "
            "CA certification required through ICAI. Work in Big 4 firms or corporate finance. "
            "Stable profession with guaranteed growth path in MNCs and public companies. "
            "Skills include accounting standards, IFRS, tax planning, financial analysis."
        ),
        "salaries": {"entry": 9, "mid": 16, "senior": 30},
        "risk": "low",
        "keywords": ["accounting", "CA", "finance", "audit", "tax", "chartered accountant"]
    },

    # --- PUBLIC SECTOR & GOVT ---
    {
        "role": "PSU Officer (GATE/Technical)",
        "category": "Government",
        "description": (
            "Work in public sector undertakings like ONGC, BHEL, NTPC, IOCL through GATE exam. "
            "Technical engineering role with government benefits, job security, pension. "
            "Highly secure job with premium entry-level pay and work-life balance. "
            "Suitable for engineering graduates seeking stability over high growth."
        ),
        "salaries": {"entry": 10, "mid": 16, "senior": 24},
        "risk": "low",
        "keywords": ["government", "PSU", "GATE", "engineering", "public sector", "stable"]
    },
    {
        "role": "Civil Services (IAS/IPS)",
        "category": "Government",
        "description": (
            "Prestigious administrative service through UPSC examination. "
            "Work as District Collector, Commissioner, policy maker in government. "
            "Immense power and social impact but extremely competitive entry. "
            "Requires years of preparation with uncertain outcome."
        ),
        "salaries": {"entry": 7, "mid": 12, "senior": 20},
        "risk": "high",
        "keywords": ["UPSC", "IAS", "IPS", "government", "civil services", "administration"]
    },
    {
        "role": "State Govt Clerk/Assistant",
        "category": "Government",
        "description": (
            "Administrative clerical work in state government departments. "
            "Handle paperwork, records, public services in government offices. "
            "Secure employment with modest compensation and pension benefits. "
            "Low stress job with work-life balance, suitable for stability seekers."
        ),
        "salaries": {"entry": 3, "mid": 5.5, "senior": 9},
        "risk": "low",
        "keywords": ["government", "clerk", "administrative", "state", "stable", "secure"]
    },

    # --- CREATIVE & OTHERS ---
    {
        "role": "Graphic Designer",
        "category": "Creative",
        "description": (
            "Create visual content using Adobe Photoshop, Illustrator, InDesign, Figma. "
            "Design logos, branding, marketing materials, UI/UX interfaces. "
            "Portfolio-based career with variable growth depending on clients. "
            "Creative role requiring artistic skills and design thinking."
        ),
        "salaries": {"entry": 3.5, "mid": 6.5, "senior": 12},
        "risk": "medium",
        "keywords": ["design", "creative", "graphics", "art", "UI", "UX", "visual"]
    },
    {
        "role": "Content Writer",
        "category": "Creative",
        "description": (
            "Write articles, blogs, website content, social media posts, copywriting. "
            "Create engaging content for brands, publications, and digital platforms. "
            "Accessible entry-level role with growth through specialization in technical "
            "writing, SEO content, or brand storytelling."
        ),
        "salaries": {"entry": 3, "mid": 6, "senior": 10},
        "risk": "medium",
        "keywords": ["writing", "content", "creative", "copywriting", "blogs", "articles"]
    },
    {
        "role": "Startup Founder / Entrepreneur",
        "category": "Business",
        "description": (
            "Start and run your own business venture. Build products, raise funding, "
            "hire teams, scale operations. High risk with potential for unlimited upside "
            "or complete failure. Requires vision, resilience, business skills, networking. "
            "Not suitable for risk-averse individuals."
        ),
        "salaries": {"entry": 0, "mid": 0, "senior": 100},
        "risk": "high",
        "keywords": ["startup", "entrepreneur", "founder", "business", "innovation", "risk"]
    },
    {
        "role": "Sales Development Rep",
        "category": "Sales",
        "description": (
            "Generate leads, cold calling, email outreach, qualify prospects for sales team. "
            "Work in B2B sales for software, services, or products companies. "
            "Performance-heavy role with commission-based compensation potential. "
            "High turnover but good path to account executive and sales management."
        ),
        "salaries": {"entry": 5, "mid": 10, "senior": 20},
        "risk": "high",
        "keywords": ["sales", "business development", "leads", "commission", "B2B"]
    }
]

# Valid input options for validation
VALID_SALARY_RANGES = ["entry", "growth", "premium"]
VALID_TIME_HORIZONS = ["immediate", "mid_term", "long_term"]
VALID_RISK_LEVELS = ["low", "medium", "high"]

# Salary range bounds in LPA
SALARY_BOUNDS = {
    "entry": (0, 6),
    "growth": (6, 12),
    "premium": (12, float("inf"))
}

# Time horizon to salary stage mapping
TIME_HORIZON_MAP = {
    "immediate": "entry",
    "mid_term": "mid",
    "long_term": "senior"
}
