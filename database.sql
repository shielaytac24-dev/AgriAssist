-- AgriAssist PostgreSQL schema
-- Create database agriassist_db first, then run this file.

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    full_name VARCHAR(120) NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(160) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    user_type VARCHAR(40) NOT NULL CHECK (user_type IN ('Farmer','Agricultural Worker','Community Member','Administrator')),
    location VARCHAR(160) NOT NULL,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until TIMESTAMP NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pests (
    id SERIAL PRIMARY KEY,
    slug VARCHAR(80) UNIQUE NOT NULL,
    common_name VARCHAR(120) NOT NULL,
    scientific_name VARCHAR(160),
    description TEXT NOT NULL,
    symptoms TEXT NOT NULL,
    damage TEXT NOT NULL,
    management TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS identifications (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    pest_id INTEGER REFERENCES pests(id) ON DELETE SET NULL,
    image_filename VARCHAR(255) NOT NULL,
    predicted_label VARCHAR(120) NOT NULL,
    confidence NUMERIC(5,2) NOT NULL CHECK (confidence >= 0 AND confidence <= 100),
    detection_count INTEGER NOT NULL DEFAULT 1,
    model_name VARCHAR(120) NOT NULL DEFAULT 'CNN-YOLOv8',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS community_posts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(180) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tips (
    id SERIAL PRIMARY KEY,
    title VARCHAR(180) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    is_published BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    title VARCHAR(180) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL DEFAULT 'INFO',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO pests(slug,common_name,scientific_name,description,symptoms,damage,management) VALUES
('brown-planthopper','Brown Planthopper','Nilaparvata lugens','A sap-sucking rice pest that can occur in rice fields and may cause hopperburn when populations become high.','Yellowing, wilting, weak plants and hopperburn.','Heavy infestations can reduce plant vigor and yield.','Scout regularly, avoid excessive nitrogen, conserve natural enemies, and follow locally recommended integrated pest management practices.'),
('green-leafhopper','Green Leafhopper','Nephotettix spp.','A green sap-sucking hopper commonly associated with rice plants.','Yellowing or reduced plant vigor and visible hopper presence.','Feeding can weaken plants and some species can contribute to disease transmission.','Monitor fields, conserve natural enemies, manage weeds and apply recommended controls only when justified.'),
('leaf-folder','Leaf Folder','Cnaphalocrocis spp.','A caterpillar that folds or rolls rice leaves and feeds within the folded leaf.','Folded leaves, scraped leaf tissue and reduced green leaf area.','Reduced photosynthetic area can weaken plants when damage is severe.','Scout regularly, conserve beneficial insects and use integrated management based on field conditions.'),
('rice-bug','Rice Bug','Leptocorisa spp.','A rice pest that can feed on developing grains.','Visible bugs around panicles and grain discoloration or damage.','Feeding can reduce grain quality and may contribute to unfilled or damaged grains.','Monitor during reproductive stages and use integrated pest management practices recommended for the locality.'),
('stem-borer','Stem Borer','Scirpophaga spp. / Chilo spp.','A larval pest group that tunnels into rice stems.','Deadhearts during vegetative growth and whiteheads during reproductive growth.','Damaged stems can reduce productive tillers and filled grains.','Use field scouting, resistant varieties where available, sanitation and integrated pest management.'),
('whorl-maggot','Whorl Maggot','Hydrellia spp.','A small fly pest whose larvae can damage young rice leaves in the whorl.','Whitish or damaged young leaves and injury in the whorl.','Severe injury can reduce leaf development and plant vigor.','Monitor young rice, maintain appropriate crop management and follow locally recommended controls.')
ON CONFLICT (slug) DO UPDATE SET common_name=EXCLUDED.common_name, scientific_name=EXCLUDED.scientific_name, description=EXCLUDED.description, symptoms=EXCLUDED.symptoms, damage=EXCLUDED.damage, management=EXCLUDED.management;

INSERT INTO tips(title,content) VALUES
('Scout Rice Fields Regularly','Inspect rice plants regularly for hopper presence, folded leaves, deadhearts, whiteheads, grain damage and unusual leaf injury.'),
('Use Integrated Pest Management','Combine monitoring, cultural practices, resistant varieties where available, biological control and properly timed interventions.'),
('Avoid Excessive Nitrogen','Follow locally recommended fertilizer rates because excessive nitrogen can produce lush growth that may favor some insect pests.'),
('Record Pest Observations','Record the date, field location, pest observed and action taken to support future farm decisions.')
ON CONFLICT (title) DO NOTHING;

INSERT INTO alerts(title,content,severity) VALUES
('Rice Pest Monitoring Reminder','Inspect rice fields regularly and document unusual pest presence or damage.', 'INFO'),
('AgriAssist Decision-Support Notice','Identification results are preliminary decision support. Confirm uncertain cases with a qualified agricultural technician or appropriate agricultural authority.', 'IMPORTANT')
ON CONFLICT (title) DO NOTHING;

-- Demo administrator. Change this password before deployment.
INSERT INTO users(full_name,username,email,password_hash,user_type,location)
VALUES ('AgriAssist Administrator','admin','admin@agriassist.local',
'pbkdf2:sha256:600000$AgriAssistDefense$962ddd1424667f3d9f261539ac84578884240a6921ed6c6318f534750d6f5a51','Administrator','System')
ON CONFLICT (username) DO NOTHING;
