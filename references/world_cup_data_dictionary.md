# World Cup Data Dictionary

## Purpose
This document defines the processed schemas and encoded value meanings for the World Cup demo datasets.

## Processed Tables

### world_cups_clean.csv
| Column | Type | Description |
|---|---|---|
| Year | str | Tournament year |
| Country | str | Host country |
| Winner | str | Champion team |
| SecondPlace | str | Runner-up team |
| ThirdPlace | str | Third-place team |
| FourthPlace | str | Fourth-place team |
| GoalsScored | int | Total goals scored in the tournament |
| QualifiedTeams | int | Number of qualified teams |
| MatchesPlayed | int | Number of matches played |
| Attendance | int | Total tournament attendance |

### world_cup_matches_clean.csv
| Column | Type | Description |
|---|---|---|
| Year | str | Tournament year |
| Datetime | datetime | Match timestamp in YYYY-MM-DD HH:MM format |
| Stage | str | Competition stage |
| Stadium | str | Match stadium |
| City | str | Host city |
| HomeTeamName | str | Home team country name |
| HomeTeamGoals | int | Goals scored by the home team |
| AwayTeamGoals | int | Goals scored by the away team |
| AwayTeamName | str | Away team country name |
| Winconditions | str | Win condition text (for example extra time, penalties) |
| Attendance | int | Match attendance |
| HalftimeHomeGoals | int | Home team goals at halftime |
| HalftimeAwayGoals | int | Away team goals at halftime |
| Referee | str (uppercase) | Match referee |
| Assistant1 | str (uppercase) | Assistant referee 1 |
| Assistant2 | str (uppercase) | Assistant referee 2 |
| RoundID | int | Tournament round identifier |
| MatchID | int | Match identifier |
| HomeTeamInitials | str (uppercase) | Home team initials |
| AwayTeamInitials | str (uppercase) | Away team initials |

### world_cup_players_clean.csv
| Column | Type | Description |
|---|---|---|
| RoundID | int | Tournament round identifier |
| MatchID | int | Match identifier |
| TeamInitials | str (uppercase) | Team initials |
| CoachName | str (uppercase) | Team coach |
| LineUp | str | Lineup flag |
| JerseyNumber | int | Player shirt number |
| PlayerName | str (uppercase) | Player name |
| PlayerPosition | str | Position code or role label |
| Event | str | Encoded event tokens for a player in a match |

## Encoded Value Definitions

### PlayerPosition
| Value | Meaning |
|---|---|
| C | Captain |
| GK | Goalkeeper |

### Event
Event may contain one or more tokens in a single value (example: G43' G87').

| Value | Meaning |
|---|---|
| G | Goal |
| OG | Own goal |
| Y | Yellow card |
| R | Red card |
| SY | Red card by second yellow |
| P | Penalty |
| MP | Missed penalty |

### LineUp
| Value | Meaning |
|---|---|
| S | Starting lineup |
| N | Substitute |

Notes:
- Composite codes such as OG, SY, and MP may appear in mixed token patterns.
- M and W appear in historical rows and should be treated as extended markers until canonical mapping is confirmed.
- Event parsing should remain token-based to support multiple events in one cell.
