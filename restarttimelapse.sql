--
-- PostgreSQL database dump
--

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;

SET search_path = public, pg_catalog;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: edgeparameters; Type: TABLE; Schema: public; Owner: igswahwsmcevan; Tablespace: 
--

CREATE TABLE edgeparameters (
    paramid integer NOT NULL,
    photoid integer,
    params character varying(150),
    rundate timestamp without time zone
);


ALTER TABLE public.edgeparameters OWNER TO igswahwsmcevan;

--
-- Name: edgeparameters_paramid_seq; Type: SEQUENCE; Schema: public; Owner: igswahwsmcevan
--

CREATE SEQUENCE edgeparameters_paramid_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.edgeparameters_paramid_seq OWNER TO igswahwsmcevan;

--
-- Name: edgeparameters_paramid_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: igswahwsmcevan
--

ALTER SEQUENCE edgeparameters_paramid_seq OWNED BY edgeparameters.paramid;


--
-- Name: edges; Type: TABLE; Schema: public; Owner: igswahwsmcevan; Tablespace: 
--

CREATE TABLE edges (
    edgeid bigint NOT NULL,
    photoid integer,
    presamelvl integer,
    nextsamelvl integer,
    child integer,
    parent integer,
    selected boolean,
    front boolean,
    geometry geometry(LineString),
    paramid integer,
    straightness1 real
);


ALTER TABLE public.edges OWNER TO igswahwsmcevan;

--
-- Name: edges_edgeid_seq; Type: SEQUENCE; Schema: public; Owner: igswahwsmcevan
--

CREATE SEQUENCE edges_edgeid_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.edges_edgeid_seq OWNER TO igswahwsmcevan;

--
-- Name: edges_edgeid_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: igswahwsmcevan
--

ALTER SEQUENCE edges_edgeid_seq OWNED BY edges.edgeid;


--
-- Name: timelapse; Type: TABLE; Schema: public; Owner: igswahwsmcevan; Tablespace: 
--

CREATE TABLE timelapse (
    photoid bigint NOT NULL,
    cameraid integer,
    locationid integer,
    date timestamp without time zone,
    image raster,
    path character varying(150)
);


ALTER TABLE public.timelapse OWNER TO igswahwsmcevan;

--
-- Name: timelapse_photoid_seq; Type: SEQUENCE; Schema: public; Owner: igswahwsmcevan
--

CREATE SEQUENCE timelapse_photoid_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.timelapse_photoid_seq OWNER TO igswahwsmcevan;

--
-- Name: timelapse_photoid_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: igswahwsmcevan
--

ALTER SEQUENCE timelapse_photoid_seq OWNED BY timelapse.photoid;


--
-- Name: paramid; Type: DEFAULT; Schema: public; Owner: igswahwsmcevan
--

ALTER TABLE ONLY edgeparameters ALTER COLUMN paramid SET DEFAULT nextval('edgeparameters_paramid_seq'::regclass);


--
-- Name: edgeid; Type: DEFAULT; Schema: public; Owner: igswahwsmcevan
--

ALTER TABLE ONLY edges ALTER COLUMN edgeid SET DEFAULT nextval('edges_edgeid_seq'::regclass);


--
-- Name: photoid; Type: DEFAULT; Schema: public; Owner: igswahwsmcevan
--

ALTER TABLE ONLY timelapse ALTER COLUMN photoid SET DEFAULT nextval('timelapse_photoid_seq'::regclass);


--
-- Name: edgeparameters_pkey; Type: CONSTRAINT; Schema: public; Owner: igswahwsmcevan; Tablespace: 
--

ALTER TABLE ONLY edgeparameters
    ADD CONSTRAINT edgeparameters_pkey PRIMARY KEY (paramid);


--
-- Name: edges_pkey; Type: CONSTRAINT; Schema: public; Owner: igswahwsmcevan; Tablespace: 
--

ALTER TABLE ONLY edges
    ADD CONSTRAINT edges_pkey PRIMARY KEY (edgeid);


--
-- Name: timelapse_pkey; Type: CONSTRAINT; Schema: public; Owner: igswahwsmcevan; Tablespace: 
--

ALTER TABLE ONLY timelapse
    ADD CONSTRAINT timelapse_pkey PRIMARY KEY (photoid);


--
-- Name: edgeparameters_photoid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: igswahwsmcevan
--

ALTER TABLE ONLY edgeparameters
    ADD CONSTRAINT edgeparameters_photoid_fkey FOREIGN KEY (photoid) REFERENCES timelapse(photoid);


--
-- Name: edges_paramid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: igswahwsmcevan
--

ALTER TABLE ONLY edges
    ADD CONSTRAINT edges_paramid_fkey FOREIGN KEY (paramid) REFERENCES edgeparameters(paramid) ON DELETE CASCADE;


--
-- Name: edges_photoid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: igswahwsmcevan
--

ALTER TABLE ONLY edges
    ADD CONSTRAINT edges_photoid_fkey FOREIGN KEY (photoid) REFERENCES timelapse(photoid);


--
-- PostgreSQL database dump complete
--

