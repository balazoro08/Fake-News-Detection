"""
Script to construct a comprehensive Fake News vs Real News dataset for model training.
Contains labeled articles spanning multiple domains (Politics, Technology, Science, Health, World, Entertainment).
"""

import os
import pandas as pd

def build_dataset():
    data = [
        # --- REAL NEWS (label = 1) ---
        {
            "title": "NASA James Webb Space Telescope Observes Oldest Galaxy Clusters in Known Universe",
            "text": "Astronomers using NASA's James Webb Space Telescope have discovered some of the earliest galaxy clusters ever observed, dating back over 13 billion years. The discovery offers crucial insights into cosmic structure formation shortly after the Big Bang. According to lead researchers at the Space Telescope Science Institute in Baltimore, spectroscopic analysis confirmed the red-shift values of these distant stellar formations. The findings have been peer-reviewed and published in the Astronomical Journal.",
            "subject": "science",
            "label": 1
        },
        {
            "title": "Federal Reserve Maintains Interest Rates Amid Moderating Inflation Data",
            "text": "The Federal Reserve voted unanimously to keep benchmark interest rates unchanged at its monthly policy meeting. Federal Reserve Chairman announced that recent economic indicators suggest steady job creation alongside gradually cooling inflation numbers. Treasury yields responded with modest declines across major indices. Economists noted that the central bank remains data-dependent before considering any future monetary easing.",
            "subject": "business",
            "label": 1
        },
        {
            "title": "World Health Organization Launches Global Initiative Against Antimicrobial Resistance",
            "text": "The World Health Organization (WHO) unveiled a coordinated global initiative aimed at mitigating rising antimicrobial resistance across clinical settings worldwide. The strategy emphasizes prudent antibiotic stewardship, expanded surveillance networks, and funding for novel therapeutic pipeline development. Public health agencies from 45 member states pledged research support and policy alignment to address drug-resistant pathogens.",
            "subject": "health",
            "label": 1
        },
        {
            "title": "European Union Reaches Landmark Agreement on Artificial Intelligence Regulation",
            "text": "European Union lawmakers successfully concluded negotiations on comprehensive legislation governing artificial intelligence systems across member nations. The regulations categorize AI applications based on risk levels, establishing strict compliance guidelines for high-risk biometric and algorithmic models. Tech Industry analysts expect the framework to set global compliance standards for AI safety and data privacy transparency.",
            "subject": "technology",
            "label": 1
        },
        {
            "title": "Global Climate Summit Concludes with Commitment to Renewable Infrastructure Funding",
            "text": "Delegates from over 190 countries concluded the annual international climate convention by finalizing a pact to triple renewable energy investments by 2030. The agreement outlines targeted funding mechanisms for developing nations transitioning away from fossil fuel reliance. Environmental scientists highlighted the critical need for rapid grid modernization to meet emission targets.",
            "subject": "world",
            "label": 1
        },
        {
            "title": "Breakthrough in Quantum Computing Achieves Sub-Nanosense Gate Error Rates",
            "text": "Researchers at Stanford University and Harvard Microelectronics Laboratory demonstrated a quantum processing architecture achieving error rates below 0.1%. Utilizing fault-tolerant logical qubits operated via neutral atom arrays, the team executed complex quantum circuits without loss of coherence. The findings were published in Nature Physics, marking a key milestone toward practical quantum advantage.",
            "subject": "technology",
            "label": 1
        },
        {
            "title": "United Nations Peacekeeping Mission Expands Humanitarian Corridor in Conflict Zone",
            "text": "The United Nations Security Council approved the expansion of humanitarian aid routes into vulnerable regions affected by recent border disputes. UN aid convoys delivered emergency medical supplies and clean water filtration units to civilian shelters. High Commissioner for Refugees commended international aid workers for securing access despite logistically challenging conditions.",
            "subject": "world",
            "label": 1
        },
        {
            "title": "FDA Approves First Targeted Gene Therapy for Rare Neuromuscular Disorder",
            "text": "The U.S. Food and Drug Administration granted regulatory approval for a novel gene therapy designed to treat spinal muscular atrophy in pediatric patients. Clinical trial data presented to the advisory panel demonstrated significant improvements in motor function scores compared to control groups. Patient advocacy groups praised the decision as a momentous step for precision medicine.",
            "subject": "health",
            "label": 1
        },
        {
            "title": "MIT Engineers Develop Ultralight Solar Cells Made from Flexible Nanomaterials",
            "text": "Engineers at the Massachusetts Institute of Technology created paper-thin solar cells capable of generating 18 watts per gram. Fabricated using scalable semiconductor inks printed onto durable fabrics, the lightweight panels can be deployed on emergency tents or portable electronics. Peer-reviewed results appear in IEEE Transactions on Nanotechnology.",
            "subject": "science",
            "label": 1
        },
        {
            "title": "Global Supply Chain Indicators Signal Stabilization Following Port Congestion Relief",
            "text": "Logistics data compiled by international shipping registries shows container throughput metrics returning to pre-pandemic baseline averages. Increased automation at major container terminals and expanded rail links reduced waiting times for commercial vessels across Asian and European trade corridors. Economists report easing freight transportation costs.",
            "subject": "business",
            "label": 1
        },
        {
            "title": "Archaeologists Unearth 3,000-Year-Old Bronze Age Settlement in Mediterranean Region",
            "text": "An international team of archaeologists uncovered intact stone structures, ceramic vessels, and bronze tools during excavations in southern Greece. Radiocarbon testing dates the settlement to approximately 1200 BCE, shedding light on maritime trade networks during the Mycenaean era. Artifacts will be preserved at the National Archaeological Museum.",
            "subject": "science",
            "label": 1
        },
        {
            "title": "Oxford University Trial Shows Efficacy of New Universal Flu Vaccine Candidate",
            "text": "Phase II clinical trial results published in The Lancet indicate a universal influenza vaccine candidate generated robust antibody responses against multiple viral strains. Scientists targeted conserved hemagglutinin stem regions to overcome annual antigenic shift challenges. Phase III trials are scheduled to commence next quarter.",
            "subject": "health",
            "label": 1
        },
        {
            "title": "International Space Station Astronauts Complete Successful Spacewalk to Repair Solar Array",
            "text": "Two NASA astronauts completed a six-hour spacewalk outside the International Space Station to install upgraded solar array modification kits. The mission successfully restored full power distribution to onboard science laboratories. Mission Control in Houston confirmed that telemetry data verified optimal array performance.",
            "subject": "science",
            "label": 1
        },
        {
            "title": "Cybersecurity Alliance Releases Open-Source Defense Tool to Protect Critical Infrastructure",
            "text": "A coalition of software security organizations released an open-source detection framework to safeguard industrial control systems against zero-day exploits. The tool monitors network telemetry for anomalous command patterns in real time. IT directors at municipal utility boards praised the community-driven defense initiative.",
            "subject": "technology",
            "label": 1
        },
        {
            "title": "Consumer Confidence Index Rises as Gas Prices Stabilize Nationally",
            "text": "The Conference Board reported a 4.2 point increase in the national Consumer Confidence Index, driven by lower retail fuel prices and stable job market data. Household spending expectations for upcoming holiday quarters showed moderate gains across major demographic categories. Financial analysts attributed consumer optimism to steady real wage growth.",
            "subject": "business",
            "label": 1
        },

        # --- FAKE NEWS (label = 0) ---
        {
            "title": "SHOCKING PROOF: Secret Government Satellite Beam Mind Control Waves Detected Across Major Cities!",
            "text": "Secret whistleblowers inside top intelligence agencies have finally leaked classified documents proving government satellites are broadcasting subliminal electromagnetic waves! They don't want you to know the truth! Millions of unsuspecting citizens are being manipulated right now to control their thoughts and buy specific products. Mainstream media is hiding this massive cover-up! Share this urgent report before it gets censored and deleted from the internet!",
            "subject": "conspiracy",
            "label": 0
        },
        {
            "title": "Doctors Are Furious! Ancient Miracle Fruit Cures All Diseases in 24 Hours, Big Pharma Panics!",
            "text": "Big Pharma is trying to ban this unbelievable natural secret! A mysterious Amazonian berry discovered by rogue researchers instantly eradicates cancer, diabetes, and heart disease overnight! Corporate drug executives are spending millions to keep this magical cure hidden from the public so they can keep profiting off expensive pills. Don't fall for their lies! Click here now to claim your secret stash before supplies run out forever!",
            "subject": "clickbait",
            "label": 0
        },
        {
            "title": "LEAKED AUDIO: Famous Celebrity Admits to Cloning Herself Three Times for Red Carpet Events!",
            "text": "An explosive audio recording obtained by anonymous hackers reveals world-famous pop star used underground military technology to clone human replicas! Sources claim the celebrity has been using genetically modified clones to perform at multiple stadium concerts simultaneously while relaxing on a private island. Hollywood insiders are terrified as the scandalous truth leaks out!",
            "subject": "entertainment_fake",
            "label": 0
        },
        {
            "title": "Scientists Confirm Earth's Core Has Completely Stopped Turning and Will Reverse Tomorrow!",
            "text": "Urgent alert! Renegade scientists warning the public that Earth's magnetic core has completely ceased rotation and is set to spin backwards starting tomorrow morning! Magnetic poles will invert, causing compasses to spin wildly and sea levels to rise instantly by 500 feet! NASA refuses to answer questions while global leaders prepare deep underground bunkers for themselves!",
            "subject": "sensational",
            "label": 0
        },
        {
            "title": "5G Towers Found to Transmit Secret Codes Controlling Weather Patterns Nationwide!",
            "text": "Bombshell discovery! An independent researcher equipped with homemade radio equipment uncovered secret frequency signals emitted by neighborhood cellular towers. According to the viral post, these 5G antennas are secretly modifying atmospheric pressure to create artificial storms and manipulate global temperatures! Share this shocking revelation with everyone you know!",
            "subject": "conspiracy",
            "label": 0
        },
        {
            "title": "Billionaire Tech Mogul Buys Moon to Build Exoskeleton Theme Park for Super-Rich",
            "text": "In an unverified viral post trending across social media, a famous tech billionaire reportedly purchased lunar mining rights from a mysterious outer-space government entity. Reports claim he plans to erect a floating luxury theme park equipped with anti-gravity rollercoasters exclusively for billionaires. Critics point out lunar ownership treaties make this completely fabricated claim impossible.",
            "subject": "satire",
            "label": 0
        },
        {
            "title": "REVEALED: Drinking Salt Water Mixed with Vinegar Instantly Boosts IQ by 50 Points!",
            "text": "A viral TikTok trend claims that drinking three glasses of concentrated salt water mixed with white vinegar every morning permanently raises human cognitive intelligence by 50 points! Medical experts warn this dangerous recipe leads to severe dehydration and kidney failure, but conspiracy channels insist establishment scientists are just suppressing high IQ secrets!",
            "subject": "health_fake",
            "label": 0
        },
        {
            "title": "Alien Spacecraft Discovered Hiding Inside Egyptian Pyramid Sealed Doorway!",
            "text": "Rogue treasure hunters breaking into a previously unopened chamber beneath the Giza plateau claim they stumbled upon an operational extraterrestrial flying saucer! Anonymous sources allege the UFO features glowing anti-gravity engines and alien hieroglyphics. Authorities allegedly confiscated all camera footage to maintain global secrecy!",
            "subject": "conspiracy",
            "label": 0
        },
        {
            "title": "Government Mandates All Citizens Replace Domestic Dogs with Synthetic Robot Pets by 2027",
            "text": "A viral blog post falsely alleges that federal regulators passed a secret bill outlawing traditional domestic canine pets in favor of state-monitored electronic robot dogs. The fake report claims citizens will be forced to turn in their golden retrievers under penalty of law. Fact-checkers verified no such legislation exists.",
            "subject": "fake_news",
            "label": 0
        },
        {
            "title": "URGENT WARNING: Tap Water Infused with Microscopic Tracking Nano-Robots!",
            "text": "Attention patriots! Whistleblower report reveals municipal water supplies are secretly loaded with micro-drones designed to track your location indoors! Filter your water immediately with special magnetic filters available only on our official store! Don't let the shadow government track your every movement!",
            "subject": "conspiracy",
            "label": 0
        },

        # --- ADDITIONAL REAL NEWS ---
        {
            "title": "Renewable Energy Capacity Surpasses Coal in Major Industrial Nations",
            "text": "Statistical agency reports show wind and solar electricity generation outpaced coal-fired energy production across major OECD countries over the past fiscal year. Rapid declines in photovoltaic module manufacturing costs coupled with government clean grid incentives accelerated the energy transition trajectory. Industry experts project continued growth in utility-scale storage installations.",
            "subject": "business",
            "label": 1
        },
        {
            "title": "Deep Sea Expedition Discovers Dozens of Previously Unknown Marine Species",
            "text": "Oceanographers deploying remotely operated underwater vehicles off the coast of South America discovered 30 new species of deep-sea corals, invertebrates, and bioluminescent fish. Operating at depths exceeding 4,000 meters, researchers recorded video footage of pristine hydrothermal vent ecosystems. Taxonomic descriptions are currently being compiled for scientific registries.",
            "subject": "science",
            "label": 1
        },
        {
            "title": "Global Literacy Rates Reach Record High Following Mobile Education Campaigns",
            "text": "UNESCO released its annual global education assessment report detailing significant improvements in youth literacy rates across sub-Saharan Africa and South Asia. The expansion of low-bandwidth digital learning platforms and community tutoring programs contributed to a 5% increase in primary school reading proficiency scores.",
            "subject": "world",
            "label": 1
        },
        {
            "title": "Automotive Industry Expands Battery Recycling Networks to Reclaim Rare Earth Metals",
            "text": "Major EV automakers partnered with recycling technology firms to establish closed-loop lithium-ion battery processing plants. The facilities utilize hydrometallurgical extraction to recover over 95% of cobalt, nickel, and lithium from depleted vehicle batteries. Environmental regulators endorsed the initiative for mitigating mining impacts.",
            "subject": "technology",
            "label": 1
        },
        {
            "title": "James Webb Telescope Detects Water Vapor in Atmosphere of Habitable-Zone Exoplanet",
            "text": "Astrophysicists analyzing transmission spectra from exoplanet K2-18b confirmed the presence of atmospheric water vapor and methane gas. Located 120 light-years away in the constellation Leo, the temperate sub-Neptune planet orbits within its host star's circumstellar habitable zone. Further spectroscopic observations are planned.",
            "subject": "science",
            "label": 1
        },

        # --- ADDITIONAL FAKE NEWS ---
        {
            "title": "UNBELIEVABLE: Man Grows Full Set of Third Teeth After Eating Raw Garlic Daily!",
            "text": "You won't believe what happened to this 60-year-old grandfather! After consuming 10 cloves of raw garlic every single morning, his gums completely regenerated and sprouted a brand new set of adult teeth overnight! Dentists are baffled and calling it a miracle! Read the full secret formula here!",
            "subject": "clickbait",
            "label": 0
        },
        {
            "title": "Secret Underground Tunnel System Found Connecting All World Capitals!",
            "text": "Conspiracy researchers allege a super-secret high-speed vacuum train network built in the 1800s connects Washington DC, London, Tokyo, and Paris deep underground! Reports claim global elites travel between continents in under 15 minutes using hyper-tech forgotten by modern history! Click here to see the secret map!",
            "subject": "conspiracy",
            "label": 0
        },
        {
            "title": "EXPOSED: Weather Bureau Admits Winter Was Cancelled by Secret Satellite Laser!",
            "text": "In a shocking video going viral online, an unverified speaker claims government lasers dissolved all snow clouds across North America! According to the fake post, winter season has been officially erased to boost tropical fruit harvests! Don't let mainstream weather reports trick you!",
            "subject": "clickbait",
            "label": 0
        },
        {
            "title": "Scientists Create Artificial Sun That Operates Without Any Energy Input!",
            "text": "Breakthrough or hoax? A blog post claims high school students invented a miniature artificial sun in their garage that produces unlimited free electricity forever without consuming any fuel or power! Power companies are allegedly sending lawyers to shut down their workshop before the grid becomes obsolete!",
            "subject": "sensational",
            "label": 0
        },
        {
            "title": "LEAKED DOCS: Instant Cell Phone Charging Hack Revealed Using Household Aluminum Foil!",
            "text": "Smart phone companies are furious after secret internal manuals revealed wrapping your smartphone in three layers of aluminum foil charges the battery from 0% to 100% in 5 seconds flat! Major tech brands don't want you to save money on chargers! Try this mind-blowing trick today!",
            "subject": "clickbait",
            "label": 0
        }
    ]

    # Expand dataset with realistic synthetic variants to ensure robust TF-IDF training
    expanded_data = list(data)
    
    # Real news templates for domain expansion
    real_topics = [
        ("Clinical Trial Demonstrates High Efficacy for Targeted Oncology Therapy", "Biotechnology researchers reported promising Phase III trial results for a novel small-molecule inhibitor targeting mutated KRAS proteins in non-small cell lung cancer. The double-blind study evaluated overall survival metrics across 1,200 participants across international medical centers. Results were published in the New England Journal of Medicine."),
        ("International Energy Agency Reports Accelerated Transition to Grid Storage Systems", "The International Energy Agency published its annual infrastructure outlook detailing unprecedented grid-scale battery deployment across North America and Europe. System operators noted improved grid reliability during peak summer demand hours due to lithium-iron-phosphate battery installations."),
        ("Central Banks Progress Toward Interoperable Cross-Border Digital Currencies", "Financial authorities from ten central banks announced successful testing of wholesale central bank digital currency (CBDC) settlement networks. The multi-currency platform reduced transaction settlement latency from days to seconds while maintaining compliance with international AML regulations."),
        ("Global Biodiversity Conservation Treaty Ratified by 120 Member Nations", "Diplomats assembled at the international environmental summit formally signed a treaty securing protected status for 30% of coastal and marine habitats by 2030. Environmental ministers agreed to standardized ecological monitoring protocols."),
        ("Robotics Institute Unveils Autonomous Drone Network for Search and Rescue Missions", "Engineers at the National Robotics Engineering Center deployed swarms of autonomous drones capable of navigating dense forest canopies without GPS signals. Utilizing thermal cameras and edge AI processing, the drones locate lost hikers in real time."),
        ("National Meteorological Center Reports Breakthrough in Early Hurricane Tracking Accuracy", "Meteorologists integrated high-resolution satellite radar data into supercomputing forecast models, extending accurate hurricane trajectory predictions from 3 days to 7 days. Emergency management officials praised the reduced false alarm rate for coastal evacuations."),
        ("Semiconductor Consortium Achieves 2-Nanometer Chip Architecture Milestone", "Leading semiconductor foundries announced commercial readiness for 2-nanometer gate-all-around nanosheet transistor manufacturing. The architecture delivers a 15% performance increase while reducing power consumption by 30% compared to previous generation nodes.")
    ]

    fake_topics = [
        ("SHOCKING: Secret Government Compound Captured on Satellite Transmitting Aliens Signals!", "Viral video sweeping conspiracy forums claims secret military radar dish is actively communicating with galactic federations! Whistleblowers claim scientists received encrypted messages containing alien technology! Mainstream news refuses to report this massive global event! Watch the censored video before it's taken down!"),
        ("Miracle Spice Burns 30 Pounds of Belly Fat in 3 Days Without Exercise!", "Dietitians are stunned! An ancient Himalayan root burns pure fat while you sleep! Corporate fitness gyms are filing lawsuits to stop people from discovering this 100% natural weight loss secret! Click to claim your free trial bottle right now!"),
        ("CONFIRMED: Tap Water Contains Chemicals Turning Garden Plants Into Spies!", "A crazy new viral post claims municipal tap water treats home houseplants with bio-sensor tracking formulas! Independent researchers claim your potted fern is secretly listening to your conversations for market research firms! Share this urgent report with all plant lovers!"),
        ("Celebrity Secretly Replaced by Android Robot After Malfunction on Live Television!", "Viewers were left terrified after famous news anchor froze for 45 seconds while her eyes flashed red on live TV broadcast! Insider sources claim the media network accidentally exposed their cybernetic android host! Executive producers claim it was just a technical glitch!"),
        ("EXPOSED: Secret Satellite Grid Turning Nighttime into Daytime Nationwide!", "Leaked government emails reveal secret plan to launch giant orbiting mirrors that reflect sunlight 24 hours a day! Sleep experts warn this will eliminate night forever! Share this shocking secret before midnight!"),
        ("Shocking Discovery: Eating Plastic Wrappers Grants Immunity to All Known Viruses!", "Dangerous social media challenge claims chewing micro-plastics protects human lungs against seasonal flu! Health organizations issued emergency warnings condemning the viral trend, but conspiracy accounts claim big pharma is just scared of free natural remedies!"),
        ("Billionaire Invents Time Machine to Travel Back to 1920s and Buy Stock Market!", "A secret tech mogul reportedly built a working temporal displacement device in an underground laboratory! Reports claim he amassed trillions by trading stocks with future knowledge! Patent offices refuse to acknowledge the invention!")
    ]

    for title, text in real_topics:
        expanded_data.append({
            "title": title,
            "text": text,
            "subject": "real_general",
            "label": 1
        })

    for title, text in fake_topics:
        expanded_data.append({
            "title": title,
            "text": text,
            "subject": "fake_general",
            "label": 0
        })

    df = pd.DataFrame(expanded_data)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/news_dataset.csv", index=False)
    print(f"Dataset created with {len(df)} articles ({df['label'].value_counts().to_dict()}). Saved to data/news_dataset.csv")

if __name__ == "__main__":
    build_dataset()
