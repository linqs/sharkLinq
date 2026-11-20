DROP TABLE IF EXISTS Proposals;
DROP TABLE IF EXISTS Actions;
DROP TABLE IF EXISTS Tokens;
DROP TABLE IF EXISTS Users;
DROP TABLE IF EXISTS Labels;
DROP TABLE IF EXISTS Tags;
DROP TABLE IF EXISTS AlternateNames;
DROP TABLE IF EXISTS Individuals;
DROP TABLE IF EXISTS Sharks;
DROP TABLE IF EXISTS Media;

CREATE TABLE Media (
    id INTEGER PRIMARY KEY,
    hash TEXT NOT NULL UNIQUE,
    name TEXT UNIQUE,
    filename TEXT UNIQUE,
    source INTEGER REFERENCES Media(id),
    metadata TEXT,
    location TEXT,
    time INTEGER,
    type TEXT,
    quality TEXT
    -- TODO who, what, copyright?
);

CREATE TABLE Individuals (
    id INTEGER PRIMARY KEY,
    name TEXT,
    sex TEXT,
    length REAL,
    media INTEGER REFERENCES Media(id), -- For individuals with media, points to associated media.
    source INTEGER REFERENCES Individuals(id), -- For individuals with media, points to the matching individual.
    reference Integer REFERENCES Individuals(id) -- Points to the reference individual, the main individual with media for a matching individual.
);

CREATE TABLE AlternateNames (
    id INTEGER PRIMARY KEY,
    individual INTEGER REFERENCES Individuals(id) ON DELETE CASCADE NOT NULL,
    name TEXT NOT NULL,
    type TEXT,
    time integer
);

CREATE TABLE Tags (
    id INTEGER PRIMARY KEY,
    individual INTEGER REFERENCES Individuals(id) ON DELETE CASCADE NOT NULL,
    name TEXT NOT NULL,
    type TEXT,
    time integer
);

CREATE TABLE Labels (
    id INTEGER PRIMARY KEY,
    individual INTEGER REFERENCES Individuals(id) ON DELETE CASCADE NOT NULL,
    name TEXT NOT NULL,
    type TEXT,
    time integer
);

CREATE TABLE Users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    username TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL,
    password TEXT NOT NULL,
    salt TEXT NOT NULL
);

CREATE TABLE Tokens (
    id INTEGER PRIMARY KEY,
    token TEXT NOT NULL,
    user INTEGER REFERENCES Users(id) ON DELETE CASCADE NOT NULL,
    expires INTEGER NOT NULL
);

CREATE TABLE Actions (
    id INTEGER PRIMARY KEY,
    original INTEGER,
    final INTEGER,
    action TEXT NOT NULL,
    object TEXT NOT NULL,
    author INTEGER REFERENCES Users (id) NOT NULL,
    time integer NOT NULL,
    notes TEXT
);

CREATE TABLE Proposals (
    id INTEGER PRIMARY KEY,
    original INTEGER,
    final INTEGER,
    action TEXT NOT NULL,
    object TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'archive')),
    author INTEGER REFERENCES Users (id) NOT NULL,
    time integer NOT NULL,
    notes TEXT
);
