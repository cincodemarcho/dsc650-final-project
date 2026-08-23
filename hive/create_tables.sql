CREATE TABLE IF NOT EXISTS cloudwatch_web_attacks (
    bytes_in BIGINT,
    bytes_out BIGINT,
    creation_time STRING,
    end_time STRING,
    src_ip STRING,
    src_ip_country_code STRING,
    protocol STRING,
    response_code INT,
    dst_port INT,
    dst_ip STRING,
    rule_names STRING,
    observation_name STRING,
    source_meta STRING,
    source_name STRING,
    `time` STRING,
    detection_types STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
TBLPROPERTIES ("skip.header.line.count"="1");
