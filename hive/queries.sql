SELECT protocol, COUNT(*), AVG(bytes_in) FROM cloudwatch_web_attacks GROUP BY protocol;
