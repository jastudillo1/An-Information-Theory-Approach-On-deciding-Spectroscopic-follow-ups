# https://skyserver.sdss.org/CasJobs/SubmitJob.aspx

DECLARE @table2 table (survey VARCHAR(30), subclass VARCHAR(30), plate SMALLINT, mjd INTEGER, fiberid SMALLINT)

DECLARE @classes VARCHAR(7) = 'OBAFGKM'
DECLARE @class VARCHAR(1)
DECLARE @i INTEGER = 1
WHILE @i <8
  BEGIN 
      SET @class = SUBSTRING(@classes,@i,1)
      INSERT INTO @table2 (survey, subclass,plate, mjd, fiberid)
      SELECT TOP 3000 survey, subclass, plate, mjd, fiberid
      FROM specObj
      WHERE class='STAR' and subclass LIKE CONCAT(@class,'%')
      SET @i = @i + 1
  END;
  
SELECT * FROM @table2